use std::{borrow::Cow, cell::RefCell};

use ndarray::{
    ArrayBase, Data, Dim, IntoDimension, Ix, RawData, RemoveAxis, SliceArg, SliceInfo,
    SliceInfoElem,
};
use num::traits::NumAssign;
use rustfft::FftNum;
use wgpu::{
    Backends, BindGroup, Buffer, ComputePipeline, Device, PipelineCompilationOptions, Queue,
    ShaderModule,
};

thread_local! {
    pub static CTX_TL: RefCell<Option<ConvGPUContext>> = const { RefCell::new(None) };
}

#[derive(Debug)]
pub struct ConvGPUContext {
    pub device: Device,
    pub queue: Queue,
    pub pipeline: ComputePipeline,
    pub bind_group_0: BindGroup,
    pub bind_group_1: BindGroup,
    pub pds_buffer: Buffer,
    pub output_buffer: Buffer,
    pub ko_buffer: Buffer,
    pub kv_buffer: Buffer,
    pub strides_buffer: Buffer,
    pub staging_buffer: Buffer,
}

impl Drop for ConvGPUContext {
    fn drop(&mut self) {
        dbg!("CTX_TL DROP!");
    }
}

async fn prepare_gpu() -> Option<(Device, Queue)> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    dbg!(instance.enumerate_adapters(Backends::VULKAN));
    dbg!(instance.enumerate_adapters(Backends::GL));

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await?;

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .ok()?;

    Some((device, queue))
}

fn prepare_cs_model(device: &Device) -> ComputePipeline {
    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("conv.wgsl"))),
    });
    // Instantiates the pipeline.
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &cs_module,
        entry_point: "main",
        compilation_options: PipelineCompilationOptions {
            zero_initialize_workgroup_memory: false,
            ..Default::default()
        },
    })
}

impl ConvGPUContext {
    fn new<'a, T, S, const N: usize>(
        device: Device,
        queue: Queue,
        pipeline: ComputePipeline,
        pds: &ArrayBase<S, Dim<[Ix; N]>>,
        offset_list: &[(isize, T)],
        output_shape: [usize; N],
    ) -> ConvGPUContext
    where
        T: NumAssign + std::fmt::Debug + FftNum,
        S: Data<Elem = T> + 'a,
        // SK: Data<Elem = T> + 'a,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
        SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>:
            SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
        Dim<[Ix; N]>: RemoveAxis,
    {
        let output_len = output_shape.iter().product::<usize>();
        // Instantiates buffer without data.
        // `usage` of buffer specifies how it can be used:
        //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
        //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (output_len * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Instantiates buffer with data (`numbers`).
        // Usage allowing the buffer to be:
        //   A storage buffer (can be bound within a bind group and thus available to a shader).
        //   The destination of a copy.
        //   The source of a copy.
        let pds_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pds Buffer"),
            size: (pds.len() * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("output Buffer"),
            size: (output_len * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let ko_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ko Buffer"),
            size: (offset_list.len() * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let kv_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv Buffer"),
            size: (offset_list.len() * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let strides_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("strides Buffer"),
            size: (N * std::mem::size_of::<[u32; 2]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: strides_buffer.as_entire_binding(),
                },
            ],
        });

        let bind_group_layout_kv = pipeline.get_bind_group_layout(1);
        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout_kv,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ko_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kv_buffer.as_entire_binding(),
                },
            ],
        });

        ConvGPUContext {
            device,
            queue,
            pipeline,
            bind_group_0,
            bind_group_1,
            pds_buffer,
            output_buffer,
            ko_buffer,
            kv_buffer,
            strides_buffer,
            staging_buffer,
        }
    }

    pub fn compute<'a, T, S, const N: usize>(
        &self,
        pds: &ArrayBase<S, Dim<[Ix; N]>>,
        offset_list: &[(isize, T)],
        strides_diff: &[[u32; 2]],
    ) where
        T: NumAssign + std::fmt::Debug + FftNum + bytemuck::Pod,
        S: Data<Elem = T> + 'a,
        // SK: Data<Elem = T> + 'a,
        [Ix; N]: IntoDimension<Dim = Dim<[Ix; N]>>,
        SliceInfo<[SliceInfoElem; N], Dim<[Ix; N]>, Dim<[Ix; N]>>:
            SliceArg<Dim<[Ix; N]>, OutDim = Dim<[Ix; N]>>,
        Dim<[Ix; N]>: RemoveAxis,
    {
        let t = std::time::Instant::now();

        self.queue.write_buffer(
            &self.pds_buffer,
            0,
            bytemuck::cast_slice(pds.as_slice().unwrap()),
        );
        self.queue.write_buffer(
            &self.ko_buffer,
            0,
            bytemuck::cast_slice(
                &offset_list
                    .iter()
                    .map(|(o, _)| *o as i32)
                    .collect::<Vec<_>>(),
            ),
        );
        self.queue.write_buffer(
            &self.kv_buffer,
            0,
            bytemuck::cast_slice(&offset_list.iter().map(|(_, v)| *v as T).collect::<Vec<_>>()),
        );
        self.queue
            .write_buffer(&self.strides_buffer, 0, bytemuck::cast_slice(strides_diff));

        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group_0, &[]);
            cpass.set_bind_group(1, &self.bind_group_1, &[]);
            cpass.insert_debug_marker("compute conv");

            let pds_shape = pds.shape().iter().map(|v| *v as u32).collect::<Vec<_>>();

            let mut rev_iter = pds_shape.iter().rev();
            let [x, y, z] = if pds_shape.len() > 3 {
                let m = pds_shape.iter().take(pds_shape.len() - 3).product::<u32>();

                let x = rev_iter.next().copied().unwrap_or(1);
                let y = rev_iter.next().copied().unwrap_or(1);
                let z = rev_iter.next().unwrap() * m;

                [x, y, z]
            } else {
                let x = rev_iter.next().copied().unwrap_or(1);
                let y = rev_iter.next().copied().unwrap_or(1);
                let z = rev_iter.next().copied().unwrap_or(1);

                [x, y, z]
            };

            // dbg!([x, y, z]);

            cpass.dispatch_workgroups(x, y, z); // Number of cells to run, the (x,y,z) size of item being processed
        }
        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.staging_buffer,
            0,
            self.output_buffer.size(),
        );

        // Submits command encoder for processing
        self.queue.submit(Some(encoder.finish()));

        let d0 = t.elapsed().as_nanos() as f64 / 1e6;

        // Note that we're not calling `.await` here.
        let buffer_slice = self.staging_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        let d1 = t.elapsed().as_nanos() as f64 / 1e6;

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        //  pollster::block_on(async {
        let d2 = t.elapsed().as_nanos() as f64 / 1e6;
        // Awaits until `buffer_future` can be read from
        if let Ok(Ok(())) = receiver.recv() {
            let d = t.elapsed().as_nanos() as f64 / 1e6;
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            self.staging_buffer.unmap(); // Unmaps buffer from memory
                                         // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                         //   delete myPointer;
                                         //   myPointer = NULL;
                                         // It effectively frees the memory

            // dbg!(&result);
            // dbg!(d_buff, d_make_pipe);
            dbg!(d0, d1, d2, d);

            // let ans_fft = data_c
            //     .map(|v| *v as f32)
            //     .conv(
            //         &kernel_c.map(|v| *v as f32),
            //         ConvMode::Same,
            //         PaddingMode::Zeros,
            //     )
            //     .unwrap()
            //     .map(|v| v.round() as i32);

            // let ans_normal = data_c
            //     .conv(&kernel_c, ConvMode::Same, PaddingMode::Zeros)
            //     .unwrap();

            // dbg!(&ans_fft, &ans_normal);

            // ans_fft
            //     .iter()
            //     .zip(result.iter())
            //     .enumerate()
            //     .for_each(|(i, (a, g))| {
            //         if (a - g) != 0 {
            //             dbg!(a, g, i);
            //             panic!();
            //         }
            //     });

            // Returns data from buffer
            // Some(result)
        } else {
            panic!("failed to run compute on gpu!")
        }
        // });
    }
}

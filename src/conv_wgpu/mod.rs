use std::borrow::Cow;

use wgpu::{ComputePipeline, Device, Queue, ShaderModule};

async fn prepare_gpu() -> Option<(Device, Queue)> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
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
        compilation_options: Default::default(),
    })
}

pub mod tests {
    use std::time::Instant;

    use ndarray::{array, Array};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use wgpu::{hal::auxil::db, util::DeviceExt};

    use crate::{dilation::IntoKernelWithDilation, padding::PaddingExt, ConvMode, PaddingMode};

    use super::*;

    // #[test]
    pub fn get_gpu() {
        let (device, queue) = pollster::block_on(prepare_gpu()).unwrap();
        let compute_pipeline = prepare_cs_model(&device);

        // let data = array![1, 2, 3, 4, 5];
        // let kernel = array![1, 1, 1];
        let data = Array::random((1000, 4000), Uniform::new(0, 100));
        let kernel = Array::random((20, 40), Uniform::new(0, 100));

        let t = Instant::now();

        let kernel = kernel.into_kernel_with_dilation();

        let explicit_conv = ConvMode::Same.unfold(&kernel);

        let pds = data.padding(PaddingMode::Zeros, explicit_conv.padding);
        let kvo = kernel.gen_offset_list(pds.strides());

        let pds = pds.into_raw_vec();

        let k_v = kvo.iter().map(|(_, v)| *v).collect::<Vec<_>>();
        let k_o = kvo.iter().map(|(o, _)| *o as i32).collect::<Vec<_>>();

        // dbg!(&pds, &kvo);

        // dbg!(&device, &queue);

        // Gets the size in bytes of the buffer.
        let size = (pds.len() * std::mem::size_of::<i32>()) as wgpu::BufferAddress;

        // dbg!(size);

        // Instantiates buffer without data.
        // `usage` of buffer specifies how it can be used:
        //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
        //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Instantiates buffer with data (`numbers`).
        // Usage allowing the buffer to be:
        //   A storage buffer (can be bound within a bind group and thus available to a shader).
        //   The destination of a copy.
        //   The source of a copy.
        let pds_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pds Buffer"),
            contents: bytemuck::cast_slice(&pds),
            usage: wgpu::BufferUsages::STORAGE
        });

        let output_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("output Buffer"),
            contents: bytemuck::cast_slice(&pds),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC,
        });

        let ko_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ko Buffer"),
            contents: bytemuck::cast_slice(&k_o),
            usage: wgpu::BufferUsages::STORAGE
        });

        let kv_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kv Buffer"),
            contents: bytemuck::cast_slice(&k_v),
            usage: wgpu::BufferUsages::STORAGE
        });

        let d_buff = t.elapsed().as_nanos() as f64 / 1e6;

        // A bind group defines how buffers are accessed by shaders.
        // It is to WebGPU what a descriptor set is to Vulkan.
        // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

        // A pipeline specifies the operation of a shader


        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
                    resource: ko_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: kv_buffer.as_entire_binding(),
                },
            ],
        });

        let d_make_pipe = t.elapsed().as_nanos() as f64 / 1e6;

        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("compute conv");

            let z = pds.len() as u32 / (65535 * 65535);
            let y = (pds.len() as u32 - z * 65535 * 65535) / 65535;
            let x = pds.len() as u32 - z * 65535 * 65535 - y * 65535;

            cpass.dispatch_workgroups(x, y.max(1) ,z.max(1)); // Number of cells to run, the (x,y,z) size of item being processed
        }
        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, size);

        // Submits command encoder for processing
        queue.submit(Some(encoder.finish()));

        let d0 = t.elapsed().as_nanos() as f64 / 1e6;

        // Note that we're not calling `.await` here.
        let buffer_slice = staging_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        let d1 = t.elapsed().as_nanos() as f64 / 1e6;

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        pollster::block_on(async {
            let d2 = t.elapsed().as_nanos() as f64 / 1e6;
            // Awaits until `buffer_future` can be read from
            if let Ok(Ok(())) = receiver.recv_async().await {
                let d = t.elapsed().as_nanos() as f64 / 1e6;
                // Gets contents of buffer
                let data = buffer_slice.get_mapped_range();
                // Since contents are got in bytes, this converts these bytes back to u32
                let result: Vec<i32> = bytemuck::cast_slice(&data).to_vec();

                // With the current interface, we have to make sure all mapped views are
                // dropped before we unmap the buffer.
                drop(data);
                staging_buffer.unmap(); // Unmaps buffer from memory
                                        // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                        //   delete myPointer;
                                        //   myPointer = NULL;
                                        // It effectively frees the memory

                // dbg!(&result);
                dbg!(d_buff, d_make_pipe);
                dbg!(d0, d1, d2, d);

                // Returns data from buffer
                Some(result)
            } else {
                panic!("failed to run compute on gpu!")
            }
        });
    }
}

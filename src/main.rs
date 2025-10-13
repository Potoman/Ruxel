#![allow(unused_imports)]
use egui_winit_vulkano::{
    Gui, GuiConfig,
    egui::{self, Color32},
};
use image::{ImageBuffer, Rgba};
use nalgebra::{Matrix4, Rotation3, Unit, Vector3, Vector4};
use std::path::PathBuf;
use std::{
    f64::consts::{FRAC_PI_2, TAU},
    fs::File,
    io::Read,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};
use vulkano::{
    Validated, Version, VulkanError, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, BlitImageInfo, ClearColorImageInfo, CommandBufferUsage,
        CopyBufferToImageInfo, CopyImageToBufferInfo, PrimaryCommandBufferAbstract,
        allocator::StandardCommandBufferAllocator,
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, DeviceOwned, Queue,
        QueueCreateInfo, QueueFlags, physical::PhysicalDeviceType,
    },
    format::{ClearColorValue, Format},
    image::{
        Image, ImageCreateInfo, ImageType, ImageUsage,
        sampler::Filter,
        view::{ImageView, ImageViewCreateInfo},
    },
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo, compute::ComputePipelineCreateInfo,
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    shader::{ShaderModule, ShaderModuleCreateInfo},
    swapchain::{
        PresentMode, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
        acquire_next_image,
    },
    sync::{self, GpuFuture},
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::KeyCode,
    window::{CursorGrabMode, Icon, Window, WindowId},
};
use winit_input_helper::WinitInputHelper;

const INITIAL_WINDOW_RESOLUTION: PhysicalSize<u32> = PhysicalSize::new(960, 960);

fn compile_to_spirv(
    path: &Path,
    kind: shaderc::ShaderKind,
    entry_point_name: &str,
) -> Result<shaderc::CompilationArtifact, shaderc::Error> {
    let mut f = File::open(path).unwrap();
    let mut source_text = String::new();
    f.read_to_string(&mut source_text).unwrap();

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    compiler.compile_into_spirv(
        &source_text,
        kind,
        path.to_str().unwrap(),
        entry_point_name,
        Some(&options),
    )
}

fn get_pipeline(shader_module: Arc<ShaderModule>) -> Arc<ComputePipeline> {
    let device = shader_module.device().clone();
    let entry_point = shader_module.entry_point("main").unwrap();

    let stage = PipelineShaderStageCreateInfo::new(entry_point);

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    ComputePipeline::new(
        device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}

fn get_allocators(
    device: &Arc<Device>,
) -> (
    Arc<StandardMemoryAllocator>,
    Arc<StandardDescriptorSetAllocator>,
    Arc<StandardCommandBufferAllocator>,
) {
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));
    (
        memory_allocator,
        descriptor_set_allocator,
        command_buffer_allocator,
    )
}

fn get_swapchain_images(
    device: &Arc<Device>,
    surface: &Arc<Surface>,
    window: &Window,
) -> (Arc<Swapchain>, Vec<Arc<Image>>) {
    let caps = device
        .physical_device()
        .surface_capabilities(surface, Default::default())
        .unwrap();

    let image_format = device
        .physical_device()
        .surface_formats(surface, Default::default())
        .unwrap()[0]
        .0;

    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();

    Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count.max(3),
            image_format,
            image_extent: window.inner_size().into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
            composite_alpha,
            present_mode: PresentMode::Immediate,
            ..Default::default()
        },
    )
    .unwrap()
}

fn get_render_image(
    memory_allocator: Arc<StandardMemoryAllocator>,
    extent: [u32; 2],
) -> (Arc<Image>, Arc<ImageView>) {
    let image = Image::new(
        memory_allocator,
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_SRC,
            format: Format::R8G8B8A8_UNORM,
            extent: [extent[0], extent[1], 1],
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
    )
    .unwrap();

    let image_view =
        ImageView::new(image.clone(), ImageViewCreateInfo::from_image(&image)).unwrap();

    (image, image_view)
}

fn get_images_and_sets(
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    render_pipeline: &ComputePipeline,
    render_extent: [u32; 2],
) -> (Arc<Image>, Arc<DescriptorSet>) {
    let (render_image, render_image_view) =
        get_render_image(memory_allocator.clone(), render_extent);

    let layout = render_pipeline.layout().set_layouts()[0].clone();
    let render_set = DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout,
        [WriteDescriptorSet::image_view(0, render_image_view.clone())],
        [],
    )
    .unwrap();

    (render_image, render_set)
}

fn get_voxel_images_and_sets(
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    render_pipeline: &ComputePipeline,
    queue: &Arc<Queue>,
    voxels: Vec<u32>,
) -> Arc<DescriptorSet> {
    let image = Image::new(
        memory_allocator.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim3d,
            format: Format::R8G8B8A8_UINT,
            extent: [4, 4, 4],
            usage: ImageUsage::STORAGE | ImageUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    let src_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        voxels,
    )
    .unwrap();

    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    command_buffer_builder
        .clear_color_image(ClearColorImageInfo::image(image.clone()))
        .unwrap()
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            src_buffer,
            image.clone(),
        ))
        .unwrap();

    let _ = command_buffer_builder
        .build()
        .unwrap()
        .execute(queue.clone())
        .unwrap();

    let image_view =
        ImageView::new(image.clone(), ImageViewCreateInfo::from_image(&image)).unwrap();

    let layout = render_pipeline
        .layout()
        .set_layouts()
        .get(1)
        .unwrap()
        .clone();
    DescriptorSet::new(
        descriptor_set_allocator.clone(),
        layout.clone(),
        [WriteDescriptorSet::image_view(0, image_view)],
        [],
    )
    .unwrap()
}

struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,

    pipeline: Arc<ComputePipeline>,

    input: WinitInputHelper,
    rcx: Option<RenderContext>,

    camera_position: Vector3<f32>,
    camera_direction: Vector3<f32>,

    is_grab: bool,
}

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    image_views: Vec<Arc<ImageView>>,

    render_image: Arc<Image>,
    render_set: Arc<DescriptorSet>,
    voxel_set: Arc<DescriptorSet>,
    gui: Gui,

    recreate_swapchain: bool,
}

impl App {
    fn new(event_loop: &EventLoop<()>) -> Self {
        // --- Instance & Surface ---
        let library = VulkanLibrary::new().unwrap();
        let mut required_extensions = Surface::required_extensions(event_loop).unwrap();

        required_extensions.ext_debug_utils = true;

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
            })
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, &event_loop).unwrap()
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .unwrap();

        // --- Logical Device ---
        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                enabled_features: DeviceFeatures {
                    dynamic_rendering: true,
                    ..DeviceFeatures::empty()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let queue = queues.next().unwrap();

        let shader_module = {
            let shader_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("shaders")
                .join("circle.comp");
            let artifact =
                compile_to_spirv(&shader_path, shaderc::ShaderKind::Compute, "main").unwrap();
            unsafe {
                ShaderModule::new(
                    device.clone(),
                    ShaderModuleCreateInfo::new(artifact.as_binary()),
                )
                .unwrap()
            }
        };

        let pipeline = get_pipeline(shader_module);

        let (memory_allocator, descriptor_set_allocator, command_buffer_allocator) =
            get_allocators(&device);

        let input = WinitInputHelper::new();

        App {
            instance,
            device,
            queue,

            memory_allocator,
            descriptor_set_allocator,
            command_buffer_allocator,

            pipeline,

            input,
            rcx: None,

            camera_position: Vector3::new(-5.0, 0.0, 0.0),
            camera_direction: Vector3::new(1.0, 0.0, 0.0),

            is_grab: false,
        }
    }

    fn render(&mut self, _event_loop: &ActiveEventLoop) {
        let rcx = self.rcx.as_mut().unwrap();

        if self.input.window_resized().is_some() {
            rcx.recreate_swapchain = true;
        }

        let window_size = rcx.window.inner_size();

        if window_size.width == 0 || window_size.height == 0 {
            return;
        }

        if rcx.recreate_swapchain {
            let images: Vec<Arc<Image>>;
            (rcx.swapchain, images) = rcx
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..rcx.swapchain.create_info()
                })
                .unwrap();

            rcx.image_views = images
                .iter()
                .map(|i| ImageView::new(i.clone(), ImageViewCreateInfo::from_image(i)).unwrap())
                .collect();

            let window_extent: [u32; 2] = window_size.into();
            (rcx.render_image, rcx.render_set) = get_images_and_sets(
                self.memory_allocator.clone(),
                self.descriptor_set_allocator.clone(),
                &self.pipeline,
                window_extent,
            );

            rcx.recreate_swapchain = false;
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    rcx.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            rcx.recreate_swapchain = true;
        }

        rcx.gui.immediate_ui(|_gui| {});

        let render_extent = rcx.render_image.extent();

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // Test
        let buf = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            (0..render_extent[0] * render_extent[1] * 4).map(|_| 0u8),
        )
        .unwrap();

        #[derive(BufferContents)]
        #[repr(C)]
        struct PushOffsetConstants {
            camera_position_x: f32,
            camera_position_y: f32,
            camera_position_z: f32,
            camera_direction_x: f32,
            camera_direction_y: f32,
            camera_direction_z: f32,
        }
        let push_offset = PushOffsetConstants {
            camera_position_x: self.camera_position.x,
            camera_position_y: self.camera_position.y,
            camera_position_z: self.camera_position.z,
            camera_direction_x: self.camera_direction.x,
            camera_direction_y: self.camera_direction.y,
            camera_direction_z: self.camera_direction.z,
        };

        builder
            .bind_pipeline_compute(self.pipeline.clone())
            .unwrap()
            .push_constants(self.pipeline.layout().clone(), 0, push_offset)
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                vec![rcx.render_set.clone(), rcx.voxel_set.clone()],
            )
            .unwrap();

        unsafe {
            builder
                .dispatch([render_extent[0], render_extent[1], 1])
                .unwrap()
                .copy_image_to_buffer(CopyImageToBufferInfo::image_buffer(
                    rcx.render_image.clone(),
                    buf.clone(),
                ))
                .unwrap();
        }

        let mut info = BlitImageInfo::images(
            rcx.render_image.clone(),
            rcx.image_views[image_index as usize].image().clone(),
        );
        info.filter = Filter::Nearest;
        builder.blit_image(info).unwrap();

        let command_buffer = builder.build().unwrap();

        // Works to debug into an image.
        //        let future = sync::now(self.device.clone())
        //            .then_execute(self.queue.clone(), command_buffer.clone())
        //            .unwrap()
        //            .then_signal_fence_and_flush()
        //            .unwrap();
        //
        //        future.wait(None).unwrap();
        //
        //        let buffer_content = buf.read().unwrap();
        //        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
        //            render_extent[0],
        //            render_extent[1],
        //            &buffer_content[..],
        //        )
        //        .unwrap();
        //        image.save("image.png").unwrap();

        let render_future = acquire_future
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap();

        let gui_future = rcx
            .gui
            .draw_on_image(render_future, rcx.image_views[image_index as usize].clone());

        gui_future
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index),
            )
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None)
            .unwrap();
    }

    fn update(&mut self, event_loop: &ActiveEventLoop) {
        if self.input.close_requested() {
            event_loop.exit();
            return;
        }
        if self.input.key_pressed(KeyCode::KeyW) || self.input.key_held(KeyCode::KeyW) {
            self.camera_position = self.camera_position + self.camera_direction * 0.1;
        }
        if self.input.key_pressed(KeyCode::KeyS) || self.input.key_held(KeyCode::KeyS) {
            self.camera_position = self.camera_position - self.camera_direction * 0.1;
        }
        if self.input.key_pressed(KeyCode::KeyA) || self.input.key_held(KeyCode::KeyA) {
            let top = Vector3::new(0.0, 0.0, 1.0);
            let dir = self.camera_position.cross(&top).normalize();
            self.camera_position -= dir * 0.1;
        }
        if self.input.key_pressed(KeyCode::KeyD) || self.input.key_held(KeyCode::KeyD) {
            let top = Vector3::new(0.0, 0.0, 1.0);
            let dir = self.camera_position.cross(&top).normalize();
            self.camera_position += dir * 0.1;
        }
        if self.input.key_pressed(KeyCode::KeyQ) || self.input.key_held(KeyCode::KeyQ) {
            self.camera_position.z -= 0.1;
        }
        if self.input.key_pressed(KeyCode::KeyE) || self.input.key_held(KeyCode::KeyE) {
            self.camera_position.z += 0.1;
        }
        if self.input.mouse_pressed(MouseButton::Left) {
            println!("mouse_pressed");
            self.is_grab = true;
        }
        if self.input.mouse_released(MouseButton::Left) {
            println!("mouse_released");
            self.is_grab = false;
        }

        if self.is_grab {
            let (dx, dy) = self.input.mouse_diff();
            // 700 -> 90 degres
            println!("mouse diff : {} {}", dx, dy);
            let rot_yaw: f32 = (90.0 * dx / 700.0).to_radians();
            let rot_pitch: f32 = (45.0 * dy / 700.0).to_radians();

            self.camera_direction =
                Rotation3::from_axis_angle(&Vector3::z_axis(), -rot_yaw) * self.camera_direction;

            let top = Vector3::new(0.0, 0.0, 1.0);
            let axis_pitch = self.camera_position.cross(&top).normalize();

            let axis_pitch_unit = Unit::new_normalize(axis_pitch);
            self.camera_direction =
                Rotation3::from_axis_angle(&axis_pitch_unit, rot_pitch) * self.camera_direction;
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_inner_size(INITIAL_WINDOW_RESOLUTION)
                        .with_title("Voxel Ray Traversal"),
                )
                .unwrap(),
        );
        let surface = Surface::from_window(self.instance.clone(), window.clone()).unwrap();

        let (swapchain, images) = get_swapchain_images(&self.device, &surface, &window);
        let image_views = images
            .iter()
            .map(|i| ImageView::new(i.clone(), ImageViewCreateInfo::from_image(i)).unwrap())
            .collect::<Vec<_>>();

        let window_extent: [u32; 2] = window.inner_size().into();
        let (render_image, render_set) = get_images_and_sets(
            self.memory_allocator.clone(),
            self.descriptor_set_allocator.clone(),
            &self.pipeline,
            window_extent,
        );

        let voxel_set = {
            let mut voxels: Vec<u32> = [0; 4 * 4 * 4].to_vec();
            voxels[0] = 1;
            voxels[1] = 0;
            voxels[2] = 0;
            voxels[3] = 0;
            voxels[4] = 0;
            voxels[5] = 0;
            voxels[6] = 0;
            voxels[7] = 0;
            voxels[8] = 0;
            voxels[9] = 0;
            voxels[10] = 0;
            voxels[11] = 0;
            voxels[12] = 1;
            voxels[13] = 0;
            voxels[14] = 0;
            voxels[15] = 1;
            voxels[12 + 16] = 1;
            voxels[12 + 16 * 2 + 3] = 1;
            voxels[12 + 16 * 3 + 3] = 1;
            voxels[12 + 16 * 3] = 1;
            get_voxel_images_and_sets(
                self.memory_allocator.clone(),
                self.command_buffer_allocator.clone(),
                self.descriptor_set_allocator.clone(),
                &self.pipeline,
                &self.queue,
                voxels,
            )
        };

        let gui = Gui::new(
            event_loop,
            surface,
            self.queue.clone(),
            swapchain.image_format(),
            GuiConfig {
                is_overlay: true,
                ..Default::default()
            },
        );

        let recreate_swapchain = false;

        self.rcx = Some(RenderContext {
            window,
            swapchain,
            image_views,

            render_image,
            render_set,
            voxel_set,
            gui,

            recreate_swapchain,
        });
    }

    fn new_events(&mut self, _event_loop: &ActiveEventLoop, _cause: winit::event::StartCause) {
        self.input.step();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if !self.rcx.as_mut().unwrap().gui.update(&event) {
            self.input.process_window_event(&event);
        }

        if event == WindowEvent::RedrawRequested {
            self.render(event_loop);
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        self.input.process_device_event(&event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.input.end_step();
        self.update(event_loop);

        let rcx = self.rcx.as_mut().unwrap();
        rcx.window.request_redraw();
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(&event_loop);
    event_loop.run_app(&mut app).unwrap();
}

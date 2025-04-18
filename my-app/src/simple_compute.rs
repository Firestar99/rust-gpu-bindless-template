#![cfg(test)]

use crate::debugger;
use approx::assert_relative_eq;
use my_app_shader::simple_compute::{add_calculation, Indirection, Param};
use pollster::block_on;
use rust_gpu_bindless::descriptor::{
	Bindless, BindlessAllocationScheme, BindlessBufferCreateInfo, BindlessBufferUsage, BindlessInstance,
	DescriptorCounts, MutDescBufferExt, RCDescExt,
};
use rust_gpu_bindless::pipeline::{HostAccess, MutBufferAccessExt, ShaderReadWrite};
use rust_gpu_bindless::platform::ash::{ash_init_single_graphics_queue, AshSingleGraphicsQueueCreateInfo};

#[test]
fn test_simple_compute_ash() -> anyhow::Result<()> {
	unsafe {
		let bindless = BindlessInstance::new(
			ash_init_single_graphics_queue(AshSingleGraphicsQueueCreateInfo {
				debug: debugger(),
				..AshSingleGraphicsQueueCreateInfo::default()
			})?,
			DescriptorCounts::REASONABLE_DEFAULTS,
		);
		block_on(test_simple_compute(&bindless))?;
		Ok(())
	}
}

async fn test_simple_compute(bindless: &Bindless) -> anyhow::Result<()> {
	let a = 42.2;
	let b = [1., 2., 3.];
	let c = 69.3;
	let len = b.len();

	// Pipelines can be created from the shaders and carry the `T` generic which is the param struct of the shader.
	let pipeline = bindless.create_compute_pipeline(crate::shader::simple_compute::simple_compute::new())?;

	// buffer_b is a slice of f32s
	let buffer_b = bindless.buffer().alloc_shared_from_iter(
		&BindlessBufferCreateInfo {
			name: "b",
			usage: BindlessBufferUsage::MAP_WRITE | BindlessBufferUsage::STORAGE_BUFFER,
			allocation_scheme: BindlessAllocationScheme::AllocatorManaged,
		},
		b,
	)?;

	// buffer_indirection holds a reference to buffer_c which contains the value c
	let buffer_indirection = {
		let buffer_c = bindless.buffer().alloc_shared_from_data(
			&BindlessBufferCreateInfo {
				name: "c",
				usage: BindlessBufferUsage::MAP_WRITE | BindlessBufferUsage::STORAGE_BUFFER,
				allocation_scheme: BindlessAllocationScheme::AllocatorManaged,
			},
			c,
		)?;
		let indirection = Indirection {
			c: buffer_c.to_strong(),
		};
		bindless.buffer().alloc_shared_from_data(
			&BindlessBufferCreateInfo {
				name: "indirection",
				usage: BindlessBufferUsage::MAP_WRITE | BindlessBufferUsage::STORAGE_BUFFER,
				allocation_scheme: BindlessAllocationScheme::AllocatorManaged,
			},
			indirection,
		)?
		// buffer_c is dropped here, but buffer_indirection having a StrongDesc on it will keep it alive for as long as
		// buffer_indirection is
	};

	let out = bindless.execute(|recording_context| {
		// b and indirection are read-only accessors to their respective buffers that only live for as long as
		// (the lifetime 'a on) recording_context does. By passing in recording_context by reference, it is ensured
		// you can't leak the accessors outside this block (apart from reentrant recording)
		let b = buffer_b.to_transient(recording_context);
		let indirection = buffer_indirection.to_transient(recording_context);

		// For mutable resources to have a safe API, they must be consumed when used in some execution, and may be converted back and returned after all commands are
		// enqueued. Here, the output buffer is created within the execution, accessed and will later be returned.
		let buffer_out = bindless
			.buffer()
			.alloc_slice(
				&BindlessBufferCreateInfo {
					name: "out",
					usage: BindlessBufferUsage::MAP_READ
						| BindlessBufferUsage::MAP_WRITE
						| BindlessBufferUsage::STORAGE_BUFFER,
					allocation_scheme: BindlessAllocationScheme::AllocatorManaged,
				},
				len,
			)
			.unwrap();
		let out = buffer_out.access::<ShaderReadWrite>(recording_context)?;

		// Enqueueing some dispatch takes in a user-supplied param struct that may contain any number of buffer
		// accesses. This method will internally "remove" the lifetime of the param struct, as the lifetime of the
		// buffers will be ensured by this execution not having finished yet.
		// Note how `a` is passed in directly, b is a reference to the buffer_b and c is not passed at all, and only accessible via indirection.
		recording_context.dispatch(
			&pipeline,
			[len as u32, 1, 1],
			Param {
				a,
				b,
				indirection,
				out: out.to_mut_transient()?,
			},
		)?;

		// you can return arbitrary data here, that can only be accessed once the execution has finished
		Ok(out.transition::<HostAccess>()?.into_desc())

		// returning makes us loose the reference on recording_context, so no accessors can leak beyond here
	})?;

	// Wait for execution to finish to map the buffer to read data from it.
	let result = out.mapped().await?.read_iter().collect::<Vec<_>>();
	let expected = b.iter().copied().map(|b| add_calculation(a, b, c)).collect::<Vec<_>>();
	println!("result: {:?}", result);
	println!("expected: {:?}", expected);
	assert_relative_eq!(&*result, &*expected, epsilon = 0.01);
	Ok(())
}

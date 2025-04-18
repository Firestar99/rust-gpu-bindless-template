use glam::UVec3;
use rust_gpu_bindless_macros::{bindless, BufferStruct};
use rust_gpu_bindless_shaders::descriptor::{Buffer, Descriptors, MutBuffer, StrongDesc, TransientDesc};

/// Deriving BufferStruct allows us to use this struct within buffer
#[derive(Copy, Clone, BufferStruct)]
pub struct Indirection {
	/// A StrongDesc allows one buffer to reference another. rust-gpu-bindless ensures the referenced buffer is not
	/// freed while Indirection is alive via internal reference counting.
	pub c: StrongDesc<Buffer<f32>>,
}

/// Our Param struct passes all the params to the shader and typically contains a single lifetime
#[derive(Copy, Clone, BufferStruct)]
pub struct Param<'a> {
	/// a value passed directly
	pub a: f32,
	/// b references a buffer containing many f32s
	/// TransientDesc have a lifetime to ensure they are only valid during a single execution (or frame).
	/// Therefore, they don't need to rely on expensive reference counting and are cheaper than a StrongDesc.
	pub b: TransientDesc<'a, Buffer<[f32]>>,
	/// b references a buffer containing a single Indirection, which in turn references a single f32 via it's c field
	pub indirection: TransientDesc<'a, Buffer<Indirection>>,
	/// a mutable buffer to write the result to
	pub out: TransientDesc<'a, MutBuffer<[f32]>>,
}

// wg of 1 is silly slow but doesn't matter
#[bindless(compute(threads(1)))]
pub fn simple_compute(
	#[bindless(descriptors)] mut descriptors: Descriptors<'_>,
	#[bindless(param)] param: &Param<'static>,
	#[spirv(workgroup_id)] wg_id: UVec3,
) {
	// read a directly
	let a = param.a;

	// read b from its buffer, using this particular index.
	// Index calculation varies widely depending on what you want to do.
	let index = wg_id.x as usize;
	let b = param.b.access(&descriptors).load(index);

	// read indirection and then read c
	let indirection = param.indirection.access(&descriptors).load();
	let c = indirection.c.access(&descriptors).load();

	// run the calculation...
	let result = add_calculation(a, b, c);

	// ...and write the result!
	// Writing the result is unsafe, as you could have multiple threads write to the same location at once.
	unsafe {
		param.out.access(&mut descriptors).store(index, result);
	}
}

pub fn add_calculation(a: f32, b: f32, c: f32) -> f32 {
	a * b + c
}

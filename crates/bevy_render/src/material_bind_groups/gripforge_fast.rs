//! GripForge (MOT-133, written here, not in Bevy 0.20.1): what the non-bindless material allocator lacked on the web
//! (every material on WebGPU and WebGL2 is non-bindless there).
//!
//! - **POD-only changes in place**: Bevy left `try_update_data` unimplemented for non-bindless materials ("TODO: Have a
//!   fast path for non-bindless materials"), so a material whose uniform data only changed was freed and allocated
//!   again: two new uniform buffers and a new bind group each time. Here, when the new bind group binds exactly the
//!   same buffers, textures, samplers and shader buffers, the data is written into the uniform buffers already there.
//!   On the web this also left two buffers to the JavaScript garbage collector per change (see
//!   `vendor/wgpu/src/backend/webgpu/gripforge_life.rs`, which now destroys them).
//! - **`write_data`**: rewrites the uniform data of a prepared material without the asset changing at all (neither
//!   re-extracted, re-prepared nor re-specialized), for a host that knows only its data moved (`MaterialParams`).
use super::{
    MaterialBindGroupAllocator, MaterialBindGroupNonBindlessAllocator, MaterialBindingId,
    MaterialNonBindlessAllocatedBindGroup,
};
use crate::{
    render_resource::{BindGroupBuilder, Buffer, OwnedBindingResource, UnpreparedBindingResource},
    renderer::RenderQueue,
};

impl MaterialBindGroupAllocator {
    /// Rewrites the uniform data of binding `binding_index` of an already prepared material in place: the bytes go to
    /// the uniform buffer its bind group holds, at [`Self::write_buffers`]. False (nothing written) for a bindless
    /// allocator, a bind group not prepared yet, or a binding of another size.
    pub fn write_data(
        &mut self,
        binding_id: MaterialBindingId,
        binding_index: u32,
        data: &[u8],
    ) -> bool {
        match *self {
            MaterialBindGroupAllocator::Bindless(_) => false,
            MaterialBindGroupAllocator::NonBindless(ref mut allocator) => {
                allocator.write_data(binding_id, binding_index, data)
            }
        }
    }
}

impl MaterialBindGroupNonBindlessAllocator {
    /// The fast path of a POD-only change (see the module). False: the slow path (free, then allocate).
    pub(super) fn try_update_data(
        &mut self,
        binding_id: MaterialBindingId,
        unprepared_bind_group: &mut BindGroupBuilder,
    ) -> bool {
        let Some(Some(MaterialNonBindlessAllocatedBindGroup::Prepared {
            bind_group,
            uniform_buffers,
            ..
        })) = self.bind_groups.get_mut(*binding_id.group as usize)
        else {
            return false;
        };
        let new = &unprepared_bind_group.binding_resources.0;
        if bind_group.bindings.len() != new.len() {
            return false;
        }
        let same = bind_group
            .bindings
            .iter()
            .zip(new.iter())
            .all(|((i, old), (j, new))| i == j && same_resource(old, new));
        let data_count = new
            .iter()
            .filter(|(_, r)| matches!(r, UnpreparedBindingResource::Data(_)))
            .count();
        if !same || data_count != uniform_buffers.len() {
            return false;
        }
        let mut buffers = uniform_buffers.iter();
        for ((_, old), (_, new)) in bind_group.bindings.iter_mut().zip(new.iter()) {
            if let (OwnedBindingResource::Data(old), UnpreparedBindingResource::Data(r)) =
                (old, new)
            {
                let buffer = buffers.next().expect("one uniform buffer per POD binding");
                let data = &unprepared_bind_group.data_buffer[(r.start as usize)..(r.end as usize)];
                stage(&mut self.pending_writes, &mut old.0, buffer, data);
            }
        }
        true
    }

    /// See [`MaterialBindGroupAllocator::write_data`].
    fn write_data(
        &mut self,
        binding_id: MaterialBindingId,
        binding_index: u32,
        data: &[u8],
    ) -> bool {
        let Some(Some(MaterialNonBindlessAllocatedBindGroup::Prepared {
            bind_group,
            uniform_buffers,
            ..
        })) = self.bind_groups.get_mut(*binding_id.group as usize)
        else {
            return false;
        };
        // The uniform buffers follow the POD bindings in order.
        let mut nth = 0;
        for (index, resource) in bind_group.bindings.iter_mut() {
            let OwnedBindingResource::Data(old) = resource else {
                continue;
            };
            if *index == binding_index {
                let Some(buffer) = uniform_buffers.get(nth) else {
                    return false;
                };
                if old.0.len() != data.len() {
                    return false;
                }
                stage(&mut self.pending_writes, &mut old.0, buffer, data);
                return true;
            }
            nth += 1;
        }
        false
    }

    /// Writes the uniform buffers rewritten this frame (from `write_buffers`).
    pub(super) fn write_pending(&mut self, render_queue: &RenderQueue) {
        for (buffer, data) in self.pending_writes.drain(..) {
            render_queue.write_buffer(&buffer, 0, &data);
        }
    }
}

/// The same GPU resource on both sides (POD: the same size).
fn same_resource(old: &OwnedBindingResource, new: &UnpreparedBindingResource) -> bool {
    match (old, new) {
        (OwnedBindingResource::Buffer(a), UnpreparedBindingResource::Buffer(b)) => a.id() == b.id(),
        (OwnedBindingResource::ShaderBuffer(a), UnpreparedBindingResource::ShaderBuffer(b)) => {
            a.id() == b.id()
        }
        (
            OwnedBindingResource::TextureView(da, a),
            UnpreparedBindingResource::TextureView(db, b),
        ) => da == db && a.id() == b.id(),
        (OwnedBindingResource::Sampler(ta, a), UnpreparedBindingResource::Sampler(tb, b)) => {
            ta == tb && a.id() == b.id()
        }
        (OwnedBindingResource::Data(a), UnpreparedBindingResource::Data(r)) => {
            a.0.len() == (r.end - r.start) as usize
        }
        _ => false,
    }
}

/// The new POD of one uniform buffer, kept (for a bind group made again later) and queued for the GPU, when it
/// changed. One write per buffer and frame: the last one wins.
fn stage(pending: &mut Vec<(Buffer, Vec<u8>)>, kept: &mut Vec<u8>, buffer: &Buffer, data: &[u8]) {
    if kept.as_slice() == data {
        return;
    }
    kept.copy_from_slice(data);
    pending.retain(|(b, _)| b.id() != buffer.id());
    pending.push((buffer.clone(), data.to_vec()));
}

#!/usr/bin/env python3
"""Test SynthStrip with FreeSurfer-exact pipeline and binary_fill_holes."""

import numpy as np
import nibabel as nib
import onnxruntime as ort
from scipy.ndimage import label, binary_fill_holes, zoom, distance_transform_edt

TOF_PATH = "/Users/uqsbollm/Downloads/testdata/tof_input.nii"
ONNX_PATH = "/Users/uqsbollm/github-repos/vesselboost-webapp/web/models/synthstrip.onnx"
OUT_DIR = "/Users/uqsbollm/Downloads/testdata"

session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
inp_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name

def run_and_analyze(vol, label_str):
    """Run inference, analyze SDT, try fill_holes."""
    # Normalize
    vmin = vol.min()
    p99 = np.percentile(vol, 99)
    norm = np.clip((vol - vmin) / (p99 - vmin + 1e-8), 0, 1).astype(np.float32)

    # Pad to 64
    shape = norm.shape
    pshape = tuple(int(np.ceil(s / 64)) * 64 for s in shape)
    padded = np.zeros(pshape, dtype=np.float32)
    padded[:shape[0], :shape[1], :shape[2]] = norm

    # Inference
    sdt = session.run([out_name], {inp_name: padded.reshape(1, 1, *pshape)})[0].squeeze()
    sdt = sdt[:shape[0], :shape[1], :shape[2]]

    print(f"\n[{label_str}] Shape: {shape}, SDT: [{sdt.min():.3f}, {sdt.max():.3f}]")

    # FreeSurfer approach: sdt < border, CC(k=1), fill=True
    for border in [0, 1]:
        mask = (sdt < border).astype(np.uint8)
        # Keep largest CC
        labeled, num = label(mask)
        if num > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            mask = (labeled == sizes.argmax()).astype(np.uint8)

        # Try fill_holes
        filled = binary_fill_holes(mask).astype(np.uint8)

        print(f"  SDT < {border}: raw={mask.sum()} -> filled={filled.sum()} "
              f"({filled.mean()*100:.1f}%) diff={filled.sum()-mask.sum()}")

        if filled.mean() > 0.15:
            print(f"    *** FILL WORKED! ***")

    return sdt

# ============ Test 1: FS-style (LIA, crop, reshape 192-320) ============
print("=" * 60)
print("TEST 1: FreeSurfer-style (LIA, crop, reshape)")
print("=" * 60)

img = nib.load(TOF_PATH)
data = img.get_fdata(dtype=np.float32)
affine = img.affine

# Reorient to LIA
lia_ornt = nib.orientations.axcodes2ornt(('L', 'I', 'A'))
orig_ornt = nib.orientations.io_orientation(affine)
xform = nib.orientations.ornt_transform(orig_ornt, lia_ornt)
lia_data = nib.orientations.apply_orientation(data, xform).astype(np.float32)

# Crop to bbox
nz = np.nonzero(lia_data > 0)
bmin = [c.min() for c in nz]
bmax = [c.max() + 1 for c in nz]
cropped = lia_data[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]].copy()
print(f"  LIA: {lia_data.shape} -> cropped: {cropped.shape}")

# Reshape to FS target
target = tuple(max(192, min(320, int(np.ceil(s / 64)) * 64)) for s in cropped.shape)
zf = tuple(t / s for t, s in zip(target, cropped.shape))
conformed = zoom(cropped, zf, order=1).astype(np.float32)
print(f"  Conformed: {conformed.shape}")

sdt1 = run_and_analyze(conformed, "FS-LIA-conform")

# ============ Test 2: RAS, pad-64, full volume ============
print("\n" + "=" * 60)
print("TEST 2: RAS, full volume, pad-64")
print("=" * 60)

ras_img = nib.as_closest_canonical(img)
ras_data = ras_img.get_fdata(dtype=np.float32)
ras_affine = ras_img.affine

sdt2 = run_and_analyze(ras_data, "RAS-full")

# ============ Test 3: RAS, crop+reshape to FS constraints ============
print("\n" + "=" * 60)
print("TEST 3: RAS, crop + reshape 192-320")
print("=" * 60)

nz3 = np.nonzero(ras_data > 0)
bmin3 = [c.min() for c in nz3]
bmax3 = [c.max() + 1 for c in nz3]
cropped3 = ras_data[bmin3[0]:bmax3[0], bmin3[1]:bmax3[1], bmin3[2]:bmax3[2]].copy()
target3 = tuple(max(192, min(320, int(np.ceil(s / 64)) * 64)) for s in cropped3.shape)
zf3 = tuple(t / s for t, s in zip(target3, cropped3.shape))
conformed3 = zoom(cropped3, zf3, order=1).astype(np.float32)
print(f"  Cropped: {cropped3.shape} -> Conformed: {conformed3.shape}")

sdt3 = run_and_analyze(conformed3, "RAS-conform")

# ============ Test 4: Try the extend_sdt approach ============
print("\n" + "=" * 60)
print("TEST 4: extend_sdt approach on FS-conformed volume")
print("=" * 60)

# Replicate FreeSurfer's extend_sdt:
# For border >= sdt.max(), recompute outer SDT from the boundary mask
sdt_test = sdt1.copy()
border_val = 1

# The SDT has narrow-band values. The "boundary" is where SDT transitions.
# The model outputs ~5 for "far from boundary" on both sides.
# extend_sdt replaces the outer part with proper EDT when border is large.

# Let's check: what if we treat the SDT differently?
# The model SDT is a narrow-band SDT: negative inside, positive outside,
# capped at ~5. For the interior far from surface, it SHOULD be large negative
# but it's +5 instead (the band is narrow).

# What if the model convention is actually INVERTED from what we expect?
# Let's check: brain interior deep should have large NEGATIVE SDT.
# But we see +5. What if we need to negate: -SDT?
neg_sdt = -sdt1
mask_neg = (neg_sdt < 1).astype(np.uint8)  # = original SDT > -1, i.e., almost everything
# That's 93%, no good.

# What if the issue is that the model WAS trained to output negative=inside
# but on TOF data specifically, it can't determine inside from outside
# for voxels far from the boundary?

# Let's try: use the SDT to find the boundary, then recompute a proper SDT
print("\nRecomputing proper SDT from boundary detection...")
boundary = (sdt1 < 1).astype(bool)  # the thin shell
print(f"  Boundary (SDT<1): {boundary.sum()} ({boundary.mean()*100:.1f}%)")

# Compute distance from boundary for ALL voxels
dist_outside = distance_transform_edt(~boundary)  # distance from boundary, for non-boundary
dist_inside_boundary = distance_transform_edt(boundary)  # not useful

# Now we need to determine sign: which side is inside brain?
# Strategy: voxels touching the volume border are exterior
# Use flood fill from corners
exterior = np.zeros_like(boundary, dtype=bool)
# Seed: volume corners
from scipy.ndimage import label as scipy_label
non_boundary = ~boundary
labeled_nb, num_nb = scipy_label(non_boundary)
print(f"  Non-boundary components: {num_nb}")

# Find which components touch the border
border_labels = set()
nx, ny, nz = sdt1.shape
for face_slice in [
    (0, slice(None), slice(None)),
    (nx-1, slice(None), slice(None)),
    (slice(None), 0, slice(None)),
    (slice(None), ny-1, slice(None)),
    (slice(None), slice(None), 0),
    (slice(None), slice(None), nz-1),
]:
    labels_on_face = labeled_nb[face_slice]
    border_labels.update(set(labels_on_face[labels_on_face > 0].ravel()))

print(f"  Border-touching components: {len(border_labels)}")

# Interior = non-boundary components NOT touching border
interior = np.zeros_like(boundary, dtype=bool)
sizes = np.bincount(labeled_nb.ravel())
for lbl in range(1, num_nb + 1):
    if lbl not in border_labels:
        interior[labeled_nb == lbl] = True
        print(f"    Interior comp {lbl}: {sizes[lbl]} voxels")

print(f"  Interior voxels: {interior.sum()}")

# Brain = boundary + interior
brain = (boundary | interior).astype(np.uint8)
brain = binary_fill_holes(brain).astype(np.uint8)
print(f"  Brain (boundary+interior+fill): {brain.sum()} ({brain.mean()*100:.1f}%)")

# Since the shell has gaps, let's try a THICKER boundary
for thresh in [2, 3, 4, 4.5, 4.8, 4.9]:
    thick_boundary = (sdt1 < thresh).astype(bool)
    non_thick = ~thick_boundary
    lb_thick, n_thick = scipy_label(non_thick)

    bl_thick = set()
    for face_slice in [
        (0, slice(None), slice(None)),
        (nx-1, slice(None), slice(None)),
        (slice(None), 0, slice(None)),
        (slice(None), ny-1, slice(None)),
        (slice(None), slice(None), 0),
        (slice(None), slice(None), nz-1),
    ]:
        labels_on_face = lb_thick[face_slice]
        bl_thick.update(set(labels_on_face[labels_on_face > 0].ravel()))

    interior_thick = np.zeros_like(thick_boundary, dtype=bool)
    sizes_thick = np.bincount(lb_thick.ravel())
    int_count = 0
    int_voxels = 0
    for lbl in range(1, n_thick + 1):
        if lbl not in bl_thick:
            interior_thick[lb_thick == lbl] = True
            int_count += 1
            int_voxels += sizes_thick[lbl]

    brain_thick = (thick_boundary | interior_thick).astype(np.uint8)
    brain_thick = binary_fill_holes(brain_thick).astype(np.uint8)
    lb_b, nb_b = scipy_label(brain_thick)
    if nb_b > 1:
        sz_b = np.bincount(lb_b.ravel()); sz_b[0] = 0
        brain_thick = (lb_b == sz_b.argmax()).astype(np.uint8)
        brain_thick = binary_fill_holes(brain_thick).astype(np.uint8)

    print(f"  SDT < {thresh}: boundary={thick_boundary.sum()}, "
          f"interior_comps={int_count} ({int_voxels}v), "
          f"final={brain_thick.sum()} ({brain_thick.mean()*100:.1f}%)")

    if brain_thick.mean() > 0.15 and brain_thick.mean() < 0.60:
        # Map back to original space and save
        inv_zf = tuple(1/z for z in zf)
        mask_unz = zoom(brain_thick.astype(np.float32), inv_zf, order=0).astype(np.uint8)
        mask_unz = mask_unz[:cropped.shape[0], :cropped.shape[1], :cropped.shape[2]]
        mask_lia = np.zeros(lia_data.shape, dtype=np.uint8)
        s = mask_unz.shape
        mask_lia[bmin[0]:bmin[0]+s[0], bmin[1]:bmin[1]+s[1], bmin[2]:bmin[2]+s[2]] = mask_unz
        inv_xform = nib.orientations.ornt_transform(lia_ornt, orig_ornt)
        mask_final = nib.orientations.apply_orientation(mask_lia, inv_xform).astype(np.uint8)

        fname = f"synthstrip_masked_t{thresh}.nii"
        nib.save(nib.Nifti1Image(data * mask_final, affine), f"{OUT_DIR}/{fname}")
        nib.save(nib.Nifti1Image(mask_final, affine, dtype=np.uint8), f"{OUT_DIR}/synthstrip_mask_t{thresh}.nii")
        print(f"    *** SAVED: {fname} ({mask_final.mean()*100:.1f}% in original space) ***")

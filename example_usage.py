import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import h5py
import fix_two_way_alignment

inFileName = 'test_stack.hdf5'
f = h5py.File(inFileName, 'r')
frames = np.copy(f['frames'])
f.close()

a = frames.mean(2)

optimal_shift, a_out= fix_two_way_alignment.subpixel(a, plot=True)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2, sharex=ax1, sharey=ax1)

ax1.imshow(a, clim=(0, 12000), cmap='gray')
ax1.set_axis_off()
ax1.set_title('Original image')

ax2.imshow(a_out, clim=(0, 12000), cmap='gray')
ax2.set_axis_off()
ax2.set_title('Aligned image')

shifted_frames = fix_two_way_alignment.shift_stack(frames.astype('float'), optimal_shift)
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2, sharex=ax1, sharey=ax1)

ax1.imshow(frames.astype('float').mean(2), clim=(0, 12000), cmap='gray')
ax1.set_axis_off()
ax1.set_title('Original image')

ax2.imshow(shifted_frames.mean(2), clim=(0, 12000), cmap='gray')
ax2.set_axis_off()
ax2.set_title('Aligned image')


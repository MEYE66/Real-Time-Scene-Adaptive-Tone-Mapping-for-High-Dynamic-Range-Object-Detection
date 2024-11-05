# File: ltm
# Description: Local Tone Mapping, perform hdr tone mapping
# Created:  上午11:16 
# Author: Gongzhe Li
import cv2
import numpy as np
from .basic_module import BasicModule
from .helpers import gammasRGB, minmax_norm



def mean_(r,g,b):
	return (r + g + b) / 3.



def meanGain_(r, g, b, k):
	# Apply gain
	rk = r * k
	gk = g * k
	bk = b * k
	# Clip the values
	rk = np.clip(rk, 0, 1.)
	gk = np.clip(gk, 0, 1.)
	bk = np.clip(bk, 0, 1.)
	# Average the channels
	return (rk + gk + bk) / 3.



def applyScaling_(initImage, shortGray, fusedGray):
	# Create a mask for division
	nonzero_mask = (shortGray != 0)

	# Calculate scaling factor s
	scaling_factor = np.ones_like(shortGray)
	scaling_factor[nonzero_mask] = fusedGray[nonzero_mask] / shortGray[nonzero_mask]

	# Apply scaling to mergedImage
	scaled_image = initImage * scaling_factor[:, :, np.newaxis]  # np.newaxis adds a new axis for broadcasting
	return scaled_image




class LTM(BasicModule):
	def __init__(self, cfg):
		super().__init__(cfg)
		self.dsFactor = 25

	def execute(self, data):
		# work with grayscale images
		image = data['rgb_image'].astype(np.uint64)
		image = np.clip(image / (self.cfg.saturation_values.hdr), 0, 1.).astype(np.float32)
		# work with shor gray
		shortGray = mean_(image[:, :, 0], image[:, :, 1], image[:, :, 2])
		shortS = cv2.resize(shortGray, (0, 0), fx=1 / self.dsFactor, fy=1 / self.dsFactor).flatten()
		bestGain = False

		gain, compression, saturated = 0, 1., 0.
		shortSg = gammasRGB(shortS, 'compress')
		sSMean = np.mean(shortSg)

		while (compression < 1.9 and saturated < .95) or (
				not (bestGain) and compression < 6 and gain < 30 and saturated < 0.33):
			gain += 2
			longSg = gammasRGB(gain * shortS, 'compress').clip(0., 1.)
			lSMean = np.mean(longSg)
			compression = lSMean / sSMean
			bestGain = lSMean > (1 - sSMean) / 2  # only works if burst underexposed
			saturated = np.sum(longSg > 0.95) / np.size(longSg)

			# print(' ----- short and long averages: ', sSMean, lSMean)
			# print(' ----- compression ratio: ', compression)
			# print(' ----- 95% saturation: ', saturated)
			# print(' ----- Automatic selection of gain = {}'.format(gain))

		# create a synthetic long exposure
		longGray = meanGain_(image[:, :, 0], image[:, :, 1], image[:, :, 2], gain)

		# apply gamma correction to both
		shortg = gammasRGB(shortGray, 'compress')
		longg = gammasRGB(longGray, 'compress')

		# perform tone mapping by exposure fusion in grayscale
		mergeMertens = cv2.createMergeMertens(contrast_weight=0., saturation_weight=0., exposure_weight=1.)

		# hack: cv2 mergeMertens expects inputs between 0 and 255
		# but the result is scaled between 0 and 1 (some values can actually be greater than 1!)
		fusedg = mergeMertens.process([255. * shortg, 255. * longg])  # .clip(0., 1.)

		# undo gamma correction
		fusedGray = gammasRGB(fusedg, 'decompress')

		# scale each RGB channel of the short exposure accordingly
		ltmImage = applyScaling_(image, shortGray, fusedGray)
		# Clip values between 0 and 1
		# ltmImage = np.clip(ltmImage, 0, 1)
		# ltmImage = fusedg
		data['rgb_image'] = ltmImage.astype(np.float32)



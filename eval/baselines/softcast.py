#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:07:52 2024

@author: khizar
"""

import numpy as np
from scipy.fft import dctn, idctn
from scipy.linalg import hadamard
import warnings
import matplotlib.pyplot as plt
import os
import cv2


def dct3(a):
    assert len(a.shape) == 3, "Expecting a 3D array!"
    return dctn(a, norm='ortho')
    

def idct3(a: np.array):
    assert len(a.shape) == 3, "Expecting a 3D array!"
    return idctn(a, norm='ortho')


class SoftCast:
    def __init__(self, ):
        self.vgen = None
        
    def encoder(self, vname, frames_per_sec, frames_per_gop, data_symbols_per_sec,
                power_budget, x_chunks = 8, y_chunks = 8):
        ## LIMITATION: something has to be done about these powers of 2!!
        ## LIMITATION: it yields zero when the video has ended!! 
        self._initialize_video(vname)
        chunks_per_frame  = int(data_symbols_per_sec / (frames_per_sec * x_chunks * y_chunks))
        chunks_per_gop    = chunks_per_frame * frames_per_gop
        while True:
            ret, gop = self.get_gop(frames_per_gop)
            if not ret: break
            dop = dct3(gop / 255) # normalize here!! 
            if dop.shape[0] % x_chunks == 0 and dop.shape[1] % y_chunks:
                warnings.warn("The image is not integer divisible into chunks!")
            x_csize = int(dop.shape[0] / x_chunks)
            y_csize = int(dop.shape[1] / y_chunks)
            
            indices, chunks = self._get_chunks_sorted(dop, x_csize, y_csize, x_chunks, 
                                                      y_chunks, frames_per_gop, chunks_per_gop)
            means, vars_, fchunks = self._he_protec(chunks, power_budget)
            
            yield True, (indices, means, vars_), fchunks
        yield False, 0, 0

    def encode_frames(self, frames, frames_per_sec, data_symbols_per_sec,
                      power_budget, x_chunks=8, y_chunks=8, channel_impulse_response=None, noise_power=None):
        """Encode video frames using SoftCast algorithm.

        Args:
            frames: Input frames as numpy array (H, W, T), values in [0, 1]
            frames_per_sec: Video frame rate
            data_symbols_per_sec: Target symbol transmission rate
            power_budget: Total transmission power budget
            x_chunks: Number of horizontal chunks (default 8)
            y_chunks: Number of vertical chunks (default 8)
            channel_impulse_response: Optional channel response for adaptive coding
            noise_power: Noise power for adaptive coding

        Returns:
            Tuple of:
                - metadata: (indices, means, vars_) tuple
                - tx_mat: Hadamard-protected data matrix, shape (chunks_per_gop, chunk_size)

        Note: The _get_chunks_sorted and _he_protec methods are already defined in the class,
        so we don't need to redefine them here. They will work with the 3D input as is.

        If channel_impulse_response is None, basic Hadamard whitening protection is applied.
        Otherwise, an adaptive video transmission design that takes the channel into account is used.
        """
        # Assuming frames are already between 0 and 1
        # Apply 3D DCT to the group of frames
        dct_frames = dct3(frames)

        # Calculate chunk sizes
        x_csize = int(dct_frames.shape[0] / x_chunks)
        y_csize = int(dct_frames.shape[1] / y_chunks)
        frames_per_gop = dct_frames.shape[2]

        # Calculate number of chunks to transmit
        chunks_per_gop = int((data_symbols_per_sec * frames_per_gop) /
                             (frames_per_sec * x_chunks * y_chunks))
        # Check if chunks_per_gop is not a power of 2
        if not (chunks_per_gop & (chunks_per_gop - 1) == 0):
            # Find the nearest lower power of 2
            chunks_per_gop = 2 ** int(np.log2(chunks_per_gop))

        # Get sorted chunks
        indices, chunks = self._get_chunks_sorted(dct_frames, x_csize, y_csize,
                                                  x_chunks, y_chunks,
                                                  frames_per_gop, chunks_per_gop)

        # Apply Hadamard Error Protection
        means, vars_, fchunks = self._he_protec(chunks, power_budget, channel_impulse_response, noise_power)

        return (indices, means, vars_), fchunks

    def encode_frames_for_channel(self, frames, frames_per_sec, data_symbols_per_sec,
                                  power_budget, x_chunks=8, y_chunks=8,
                                  output_dtype=np.float64):
        """Encode frames and prepare for channel transmission via I/Q modulation.

        This is a convenience wrapper around encode_frames() that ensures the output
        is in the correct format for the channel simulation pipeline.

        Args:
            frames: Input frames as numpy array (H, W, T), values in [0, 1]
            frames_per_sec: Video frame rate
            data_symbols_per_sec: Target symbol transmission rate
            power_budget: Total transmission power budget
            x_chunks: Number of horizontal chunks (default 8)
            y_chunks: Number of vertical chunks (default 8)
            output_dtype: Output data type for tx_mat (default np.float64)

        Returns:
            Tuple of:
                - metadata: (indices, means, vars_) tuple where:
                    - indices: List of (i, j, k) tuples indicating chunk positions
                    - means: numpy array of chunk means (float32)
                    - vars_: numpy array of chunk variances (float32)
                - tx_mat: Real-valued transmission matrix, shape (chunks_per_gop, chunk_size)
                         Ready for I/Q modulation (pairs of values -> complex)
                - info: Dict with encoding parameters:
                    - chunks_per_gop: Number of chunks transmitted
                    - chunk_size: Size of each chunk (flattened)
                    - total_samples: Total real samples (tx_mat.size)
        """
        metadata, tx_mat = self.encode_frames(
            frames=frames,
            frames_per_sec=frames_per_sec,
            data_symbols_per_sec=data_symbols_per_sec,
            power_budget=power_budget,
            x_chunks=x_chunks,
            y_chunks=y_chunks,
        )

        indices, means, vars_ = metadata

        # Ensure consistent dtypes
        means = np.asarray(means, dtype=np.float32)
        vars_ = np.asarray(vars_, dtype=np.float32)
        tx_mat = np.asarray(tx_mat, dtype=output_dtype)

        info = {
            'chunks_per_gop': tx_mat.shape[0],
            'chunk_size': tx_mat.shape[1] if tx_mat.ndim > 1 else tx_mat.size,
            'total_samples': tx_mat.size,
            'frames_per_gop': frames.shape[2] if frames.ndim == 3 else 1,
        }

        return (indices, means, vars_), tx_mat, info

    def _he_protec(self, chunks, power_budget, channel_impulse_response=None, noise_power=None):
        # error protection and resilience to packet loss
        mat_chunks = np.stack(chunks, axis = 0).reshape((len(chunks), -1))
        means = np.mean(mat_chunks, axis = 1)
        mat_chunks = mat_chunks - np.tile(means, (mat_chunks.shape[1], 1)).T
        vars_ = np.var(mat_chunks, axis = 1)
        if channel_impulse_response is None:
            gains = np.diag(np.power(vars_, -0.25) * np.sqrt(power_budget / np.sum(np.sqrt(vars_))))
            mat_hadamard = hadamard(gains.shape[0])
            tx_mat = np.matmul(np.matmul(mat_hadamard, gains), mat_chunks)
        else:
            # TODO: implement the adaptive video transmission design
            # that takes into account the channel impulse response
            # Note: Here be dragons!
            H = channel_impulse_response.T
            H = H[:H.shape[1], :]

            # Perform eigen-decomposition of the channel impulse response
            Lambda, V = np.linalg.eig(1/noise_power * np.conjugate(H.T) @ H)
            Lambda = np.abs(Lambda)
            lambda_mtx = np.diag(Lambda)

            # since the chunks are sorted, the permutation matrix is the identity matrix
            # R_d is the covariance matrix of the data, we only need the diagonal elements, which is vars_
            mu = 0
            lambda_inv_sum = 0
            for i in range(Lambda.size):
                mu += np.sqrt(vars_[i]) / np.sqrt(Lambda[i])
                lambda_inv_sum += 1 / Lambda[i]
            mu = mu / (lambda_inv_sum + power_budget)

            gamma = np.zeros_like(Lambda)
            for i in range(Lambda.size):
                gamma[i] = 1/np.sqrt(mu * vars_[i] * Lambda[i]) - 1/(vars_[i] * Lambda[i])
            
            phi = np.zeros_like(Lambda)
            for i in range(Lambda.size):
                # and gamma[i] < power_budget/np.sqrt(noise_power)
                phi[i] = np.sqrt(gamma[i]) if gamma[i] > 0 else 0
            alpha = np.sum([(phi[i] ** 2) * vars_[i] for i in range(Lambda.size)])
            
        return means, vars_, tx_mat
    
    def _get_chunks_sorted(self, dop, x_csize, y_csize, x_chunks, y_chunks, 
                           frames_per_gop, chunks_per_gop):
        indices = []
        sum_of_squares = []
        chunks = []
        for i in range(x_csize):
            for j in range(y_csize):
                for k in range(frames_per_gop):
                    indices.append((i, j, k))
                    chunks.append(dop[i*x_chunks:(i+1)*x_chunks,
                                      j*x_chunks:(j+1)*y_chunks,
                                      k])
                    sum_of_squares.append(np.sum(np.square(chunks[-1]), axis = None))
        sorted_inds = np.flip(np.argsort(sum_of_squares))
        chunks[:]   = [chunks[i] for i in sorted_inds][:chunks_per_gop]
        indices[:]  = [indices[i] for i in sorted_inds][:chunks_per_gop]
        return indices, chunks

    def _frame_generator(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                yield False, None
            else:
                # Convert BGR to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                yield True, gray

    def _initialize_video(self, vname): 
        self.cap = cv2.VideoCapture(vname)
        self.vgen = self._frame_generator()
        
    def get_gop(self, frames_per_gop):
        lop = []
        ret = True
        for i in range(frames_per_gop):
            ret, img = next(self.vgen)
            if ret:
                cutoff = int(img.shape[0] * 2 / 3)
                y = img[:cutoff ,  :]
                lop.append(y)
            else: 
                return False, 0
        return ret, np.stack(lop, axis = -1) # get the group of pictures!
    
    def decode(self, metadata, data, coding_noises, frames_per_gop, 
                power_budget, x_chunks = 8, y_chunks = 8,
                x_vid = 144, y_vid = 176):
        indices = metadata[0]
        means   = metadata[1]
        vars_   = metadata[2]
        gains = np.diag(np.power(vars_, -0.25) * np.sqrt(power_budget / np.sum(np.sqrt(vars_))))
        coding_mat = hadamard(gains.shape[0]) @ gains
        dvars = np.diag(vars_)
        x_llse = dvars @ coding_mat.T @ np.linalg.inv(coding_mat \
                      @ dvars @ coding_mat.T + coding_noises) @ data
        # x_llse = np.linalg.inv(coding_mat) @ data
        x_llse = x_llse + np.tile(means, (x_llse.shape[1], 1)).T
        rec_dop = np.zeros((x_vid, y_vid, frames_per_gop))
        for i, ind in enumerate(indices):
            for j, slice_ in enumerate(x_llse[i]):
                rec_dop[y_chunks * ind[0] + int(j / x_chunks), x_chunks * ind[1] + j % x_chunks, ind[2]] = slice_
        rec_gop = idct3(rec_dop)
        return rec_gop
                    
        
if __name__ == "__main__":
    pass
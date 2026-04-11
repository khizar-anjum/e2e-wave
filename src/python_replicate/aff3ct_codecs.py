"""
AFF3CT-based Forward Error Correction Codecs

Provides high-performance FEC codec implementations using the py_aff3ct library,
including LDPC, Polar, Turbo, and Convolutional (RSC) codes with soft-decision decoding.

Requires py_aff3ct to be built and available in the Python path.
See memory/AFF3CT_API.md for API documentation.
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np

# Add py_aff3ct build directory to path
_py_aff3ct_path = Path(__file__).parent.parent / 'py_aff3ct' / 'build' / 'lib'
if _py_aff3ct_path.exists():
    sys.path.insert(0, str(_py_aff3ct_path))

from python_replicate.fec_codec import FECCodec


class AFF3CTCodec(FECCodec):
    """Base class for py_aff3ct-based codecs.

    All AFF3CT codecs support soft-decision (LLR) decoding for improved
    error correction performance.
    """

    def __init__(self, k: int, n: int):
        """Initialize AFF3CT codec.

        Args:
            k: Message length (info bits)
            n: Codeword length (coded bits)

        Raises:
            ImportError: If py_aff3ct is not available
        """
        try:
            import py_aff3ct as aff3ct
            self._aff3ct = aff3ct
        except ImportError as e:
            raise ImportError(
                "py_aff3ct not found. Ensure it is built at py_aff3ct/build/lib/. "
                f"Original error: {e}"
            )
        self._k = k
        self._n = n
        self._encoder = None
        self._decoder = None

    def supports_soft_decoding(self) -> bool:
        """AFF3CT codecs support soft-decision decoding."""
        return True

    @property
    def k(self) -> int:
        """Message length in bits."""
        return self._k

    @property
    def n(self) -> int:
        """Codeword length in bits."""
        return self._n

    @property
    def rate(self) -> float:
        """Code rate (k/n)."""
        return self._k / self._n


class LDPCCodec(AFF3CTCodec):
    """LDPC codec using Belief Propagation decoding.

    Supports loading LDPC codes from .alist or .qc (quasi-cyclic) files.
    Pre-configured codes are available in py_aff3ct/lib/aff3ct/conf/dec/LDPC/.
    """

    def __init__(self, alist_path: str, max_iter: int = 50):
        """Initialize LDPC codec from parity check matrix file.

        Args:
            alist_path: Path to .alist or .qc file defining the LDPC code
            max_iter: Maximum BP decoder iterations (default: 50)

        Example:
            codec = LDPCCodec("py_aff3ct/lib/aff3ct/conf/dec/LDPC/CCSDS_64_128.alist")
        """
        # Load parity check matrix first to determine dimensions
        import py_aff3ct.tools.sparse_matrix as sp

        alist_path = str(alist_path)
        if alist_path.endswith('.qc'):
            H = sp.qc.read(alist_path)
        else:
            H = sp.alist.read(alist_path)

        # H is (N, M) where N = codeword length, M = parity bits
        n = H.shape[0]
        m = H.shape[1]
        k = n - m

        super().__init__(k, n)
        self._H = H
        self._max_iter = max_iter
        self._alist_path = alist_path

        # Create encoder and decoder
        self._encoder = self._aff3ct.module.encoder.Encoder_LDPC_from_QC(k, n, H)
        info_pos = self._encoder.get_info_bits_pos()
        self._decoder = self._aff3ct.module.decoder.Decoder_LDPC_BP_horizontal_layered_inter_NMS(
            k, n, max_iter, H, info_pos
        )

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encode message bits to LDPC codeword.

        Args:
            bits: 1D array of message bits (will be padded to multiple of K)

        Returns:
            Encoded codeword bits
        """
        bits = np.asarray(bits, dtype=np.int32).flatten()

        # Pad to multiple of K
        n_blocks = int(np.ceil(len(bits) / self._k))
        padded = np.zeros(n_blocks * self._k, dtype=np.int32)
        padded[:len(bits)] = bits

        # Encode each block
        encoded = []
        for i in range(n_blocks):
            block = padded[i * self._k : (i + 1) * self._k]
            # AFF3CT expects 2D array with shape (1, K)
            block_2d = block.reshape(1, -1)
            self._encoder["encode::U_K"].bind(block_2d)
            self._encoder["encode"].exec()
            encoded.append(self._encoder["encode::X_N"][:].flatten().copy())

        return np.concatenate(encoded).astype(np.uint8)

    def decode_soft(self, llrs: np.ndarray) -> np.ndarray:
        """Decode using log-likelihood ratios (soft-decision).

        Args:
            llrs: 1D array of LLRs (positive = more likely bit 0)

        Returns:
            Decoded message bits
        """
        llrs = np.asarray(llrs, dtype=np.float32).flatten()
        n_blocks = len(llrs) // self._n

        decoded = []
        for i in range(n_blocks):
            block = llrs[i * self._n : (i + 1) * self._n]
            # AFF3CT expects 2D array with shape (1, N)
            block_2d = block.reshape(1, -1)
            self._decoder["decode_siho::Y_N"].bind(block_2d)
            self._decoder["decode_siho"].exec()
            decoded.append(self._decoder["decode_siho::V_K"][:].flatten().copy())

        return np.concatenate(decoded).astype(np.uint8)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        """Decode using hard-decision bits (converts to LLRs internally).

        Args:
            bits: 1D array of received bits (0 or 1)

        Returns:
            Decoded message bits
        """
        # Convert hard bits to LLRs: 0 -> +1 (likely 0), 1 -> -1 (likely 1)
        bits = np.asarray(bits, dtype=np.float32).flatten()
        llrs = 1.0 - 2.0 * bits
        return self.decode_soft(llrs)


class DVBS2LDPCCodec(AFF3CTCodec):
    """DVB-S2 LDPC codec with variable code rates.

    Supports code rates from 0.2 (1/5) to 0.9 using the DVB-S2 standard
    LDPC matrices. This is the best option for LDPC codes at low rates.

    Available rates:
        Short frame (N=16200): 0.20, 0.33, 0.40, 0.44, 0.60, 0.67, 0.73, 0.78, 0.82, 0.89
        Normal frame (N=64800): 0.25, 0.33, 0.40, 0.50, 0.60, 0.67, 0.75, 0.80, 0.83, 0.89, 0.90
    """

    # Available DVB-S2 codes: (N, M) -> rate, where K = N - M
    AVAILABLE_CODES = {
        # Short frame (N=16200)
        (16200, 12960): 0.20,   # K=3240
        (16200, 10800): 0.33,   # K=5400
        (16200, 9720): 0.40,    # K=6480
        (16200, 9000): 0.44,    # K=7200
        (16200, 6480): 0.60,    # K=9720
        (16200, 5400): 0.67,    # K=10800
        (16200, 4320): 0.73,    # K=11880
        (16200, 3600): 0.78,    # K=12600
        (16200, 2880): 0.82,    # K=13320
        (16200, 1800): 0.89,    # K=14400
        # Normal frame (N=64800)
        (64800, 48600): 0.25,   # K=16200
        (64800, 43200): 0.33,   # K=21600
        (64800, 38880): 0.40,   # K=25920
        (64800, 32400): 0.50,   # K=32400
        (64800, 25920): 0.60,   # K=38880
        (64800, 21600): 0.67,   # K=43200
        (64800, 16200): 0.75,   # K=48600
        (64800, 12960): 0.80,   # K=51840
        (64800, 10800): 0.83,   # K=54000
        (64800, 7200): 0.89,    # K=57600
        (64800, 6480): 0.90,    # K=58320
    }

    @classmethod
    def available_rates(cls, frame: str = 'both') -> list:
        """Return list of available code rates.

        Args:
            frame: 'short' (N=16200), 'normal' (N=64800), or 'both'

        Returns:
            List of (rate, K, N) tuples sorted by rate
        """
        rates = []
        for (n, m), rate in cls.AVAILABLE_CODES.items():
            if frame == 'short' and n != 16200:
                continue
            if frame == 'normal' and n != 64800:
                continue
            k = n - m
            rates.append((rate, k, n))
        return sorted(rates, key=lambda x: x[0])

    @classmethod
    def find_code(cls, target_rate: float, frame: str = 'short') -> tuple:
        """Find the closest available code to target rate.

        Args:
            target_rate: Desired code rate (0.0 to 1.0)
            frame: 'short' (N=16200) or 'normal' (N=64800)

        Returns:
            Tuple of (actual_rate, K, N)
        """
        n_filter = 16200 if frame == 'short' else 64800
        candidates = [(rate, n - m, n) for (n, m), rate in cls.AVAILABLE_CODES.items() if n == n_filter]
        if not candidates:
            raise ValueError(f"No codes available for frame type '{frame}'")
        return min(candidates, key=lambda x: abs(x[0] - target_rate))

    def __init__(self, k: int, n: int, max_iter: int = 50):
        """Initialize DVB-S2 LDPC codec.

        Args:
            k: Message length (info bits)
            n: Codeword length (16200 or 64800)
            max_iter: Maximum BP decoder iterations (default: 50)

        Example:
            # Rate 1/5 (0.20) using short frame
            codec = DVBS2LDPCCodec(k=3240, n=16200)

            # Rate 1/4 (0.25) using normal frame
            codec = DVBS2LDPCCodec(k=16200, n=64800)

            # Find and use closest rate to target
            rate, k, n = DVBS2LDPCCodec.find_code(0.25, frame='short')
            codec = DVBS2LDPCCodec(k=k, n=n)
        """
        # Calculate M (parity bits) from K and N
        m = n - k

        # Verify this is a valid DVB-S2 code
        if (n, m) not in self.AVAILABLE_CODES:
            available = [(n - m, n, rate) for (n, m), rate in self.AVAILABLE_CODES.items() if n == n]
            raise ValueError(
                f"No DVB-S2 code with K={k}, N={n}. "
                f"Available codes for N={n}: {available}"
            )

        super().__init__(k, n)
        self._max_iter = max_iter

        # Get the DVB-S2 values class
        dvbs2_class_name = f"dvbs2_values_{n}_{m}"
        dvbs2_class = getattr(self._aff3ct.tools.dvbs2_values, dvbs2_class_name)
        self._dvbs2_val = dvbs2_class()

        # Build H matrix
        self._H = self._dvbs2_val.build_H()

        # Create encoder and decoder
        self._encoder = self._aff3ct.module.encoder.Encoder_LDPC_DVBS2(self._dvbs2_val)
        info_pos = self._encoder.get_info_bits_pos()
        self._decoder = self._aff3ct.module.decoder.Decoder_LDPC_BP_horizontal_layered_inter_NMS(
            k, n, max_iter, self._H, info_pos
        )

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encode message bits to LDPC codeword."""
        bits = np.asarray(bits, dtype=np.int32).flatten()

        # Pad to multiple of K
        n_blocks = int(np.ceil(len(bits) / self._k))
        padded = np.zeros(n_blocks * self._k, dtype=np.int32)
        padded[:len(bits)] = bits

        # Encode each block
        encoded = []
        for i in range(n_blocks):
            block = padded[i * self._k : (i + 1) * self._k]
            block_2d = block.reshape(1, -1)
            self._encoder["encode::U_K"].bind(block_2d)
            self._encoder["encode"].exec()
            encoded.append(self._encoder["encode::X_N"][:].flatten().copy())

        return np.concatenate(encoded).astype(np.uint8)

    def decode_soft(self, llrs: np.ndarray) -> np.ndarray:
        """Decode using log-likelihood ratios (soft-decision)."""
        llrs = np.asarray(llrs, dtype=np.float32).flatten()
        n_blocks = len(llrs) // self._n

        decoded = []
        for i in range(n_blocks):
            block = llrs[i * self._n : (i + 1) * self._n]
            block_2d = block.reshape(1, -1)
            self._decoder["decode_siho::Y_N"].bind(block_2d)
            self._decoder["decode_siho"].exec()
            decoded.append(self._decoder["decode_siho::V_K"][:].flatten().copy())

        return np.concatenate(decoded).astype(np.uint8)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        """Decode using hard-decision bits."""
        bits = np.asarray(bits, dtype=np.float32).flatten()
        llrs = 1.0 - 2.0 * bits
        return self.decode_soft(llrs)


class PolarCodec(AFF3CTCodec):
    """Polar codec using Successive Cancellation (SC) decoding.

    Polar codes are capacity-achieving codes used in 5G NR control channels.
    The frozen bit positions are computed using Gaussian Approximation.
    """

    def __init__(self, k: int, n: int, sigma: float = 0.5):
        """Initialize Polar codec.

        Args:
            k: Message length (info bits)
            n: Codeword length (must be power of 2)
            sigma: Design noise level for frozen bit selection (default: 0.5)
        """
        if n & (n - 1) != 0:
            raise ValueError(f"Codeword length n={n} must be a power of 2")
        if k > n:
            raise ValueError(f"Message length k={k} cannot exceed codeword length n={n}")

        super().__init__(k, n)
        self._sigma = sigma

        # Generate frozen bits using Gaussian Approximation
        fbgen = self._aff3ct.tools.frozenbits_generator.Frozenbits_generator_GA_Arikan(k, n)
        noise = self._aff3ct.tools.noise.Sigma(sigma)
        fbgen.set_noise(noise)
        self._frozen_bits = fbgen.generate()

        # Create encoder and decoder
        self._encoder = self._aff3ct.module.encoder.Encoder_polar_sys(k, n, self._frozen_bits)
        self._decoder = self._aff3ct.module.decoder.Decoder_polar_SC_fast_sys(k, n, self._frozen_bits)

    def update_frozen_bits(self, sigma: float):
        """Update frozen bit positions for a new noise level.

        For optimal performance, frozen bits should be updated when the
        operating SNR changes significantly.

        Args:
            sigma: New noise level (standard deviation)
        """
        self._sigma = sigma
        fbgen = self._aff3ct.tools.frozenbits_generator.Frozenbits_generator_GA_Arikan(
            self._k, self._n
        )
        noise = self._aff3ct.tools.noise.Sigma(sigma)
        fbgen.set_noise(noise)
        self._frozen_bits = fbgen.generate()
        self._encoder.set_frozen_bits(self._frozen_bits)
        self._decoder.set_frozen_bits(self._frozen_bits)

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encode message bits to Polar codeword.

        Args:
            bits: 1D array of message bits (will be padded to multiple of K)

        Returns:
            Encoded codeword bits
        """
        bits = np.asarray(bits, dtype=np.int32).flatten()

        # Pad to multiple of K
        n_blocks = int(np.ceil(len(bits) / self._k))
        padded = np.zeros(n_blocks * self._k, dtype=np.int32)
        padded[:len(bits)] = bits

        # Encode each block
        encoded = []
        for i in range(n_blocks):
            block = padded[i * self._k : (i + 1) * self._k]
            # AFF3CT expects 2D array with shape (1, K)
            block_2d = block.reshape(1, -1)
            self._encoder["encode::U_K"].bind(block_2d)
            self._encoder["encode"].exec()
            encoded.append(self._encoder["encode::X_N"][:].flatten().copy())

        return np.concatenate(encoded).astype(np.uint8)

    def decode_soft(self, llrs: np.ndarray) -> np.ndarray:
        """Decode using log-likelihood ratios (soft-decision).

        Note: Polar decoder outputs signed values where negative = bit 1,
        non-negative = bit 0. This method handles the conversion automatically.

        Args:
            llrs: 1D array of LLRs (positive = more likely bit 0)

        Returns:
            Decoded message bits
        """
        llrs = np.asarray(llrs, dtype=np.float32).flatten()
        n_blocks = len(llrs) // self._n

        decoded = []
        for i in range(n_blocks):
            block = llrs[i * self._n : (i + 1) * self._n]
            # AFF3CT expects 2D array with shape (1, N)
            block_2d = block.reshape(1, -1)
            self._decoder["decode_siho::Y_N"].bind(block_2d)
            self._decoder["decode_siho"].exec()
            raw_output = self._decoder["decode_siho::V_K"][:].flatten().copy()
            # Polar decoder outputs signed values: negative = 1, non-negative = 0
            decoded.append((raw_output < 0).astype(np.uint8))

        return np.concatenate(decoded)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        """Decode using hard-decision bits."""
        bits = np.asarray(bits, dtype=np.float32).flatten()
        llrs = 1.0 - 2.0 * bits
        return self.decode_soft(llrs)


class RSCCodec(AFF3CTCodec):
    """Recursive Systematic Convolutional (RSC) codec with BCJR decoding.

    RSC codes are the building blocks for Turbo codes. This implementation
    uses the BCJR (MAP) algorithm for soft-decision decoding.
    """

    def __init__(self, k: int):
        """Initialize RSC codec.

        Args:
            k: Message length (info bits)

        Note:
            Codeword length is N = 2*K + 4 (rate ~1/2 with tail bits).
            Example: K=64 -> N=132
        """
        # RSC has rate ~1/2, N = 2*K + 4 (systematic + parity + tail bits)
        n = 2 * k + 4

        super().__init__(k, n)

        # Create encoder and decoder
        self._encoder = self._aff3ct.module.encoder.Encoder_RSC_generic_sys(k, n)

        # Get trellis for decoder
        trellis = self._encoder.get_trellis()
        self._decoder = self._aff3ct.module.decoder.Decoder_RSC_BCJR_seq_generic_std(k, trellis)

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encode message bits to RSC codeword.

        Args:
            bits: 1D array of message bits (will be padded to multiple of K)

        Returns:
            Encoded codeword bits
        """
        bits = np.asarray(bits, dtype=np.int32).flatten()

        # Pad to multiple of K
        n_blocks = int(np.ceil(len(bits) / self._k))
        padded = np.zeros(n_blocks * self._k, dtype=np.int32)
        padded[:len(bits)] = bits

        # Encode each block
        encoded = []
        for i in range(n_blocks):
            block = padded[i * self._k : (i + 1) * self._k]
            # AFF3CT expects 2D array with shape (1, K)
            block_2d = block.reshape(1, -1)
            self._encoder["encode::U_K"].bind(block_2d)
            self._encoder["encode"].exec()
            encoded.append(self._encoder["encode::X_N"][:].flatten().copy())

        return np.concatenate(encoded).astype(np.uint8)

    def decode_soft(self, llrs: np.ndarray) -> np.ndarray:
        """Decode using log-likelihood ratios (soft-decision).

        Args:
            llrs: 1D array of LLRs (positive = more likely bit 0)

        Returns:
            Decoded message bits
        """
        llrs = np.asarray(llrs, dtype=np.float32).flatten()
        n_blocks = len(llrs) // self._n

        decoded = []
        for i in range(n_blocks):
            block = llrs[i * self._n : (i + 1) * self._n]
            # AFF3CT expects 2D array with shape (1, N)
            block_2d = block.reshape(1, -1)
            self._decoder["decode_siho::Y_N"].bind(block_2d)
            self._decoder["decode_siho"].exec()
            decoded.append(self._decoder["decode_siho::V_K"][:].flatten().copy())

        return np.concatenate(decoded).astype(np.uint8)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        """Decode using hard-decision bits."""
        bits = np.asarray(bits, dtype=np.float32).flatten()
        llrs = 1.0 - 2.0 * bits
        return self.decode_soft(llrs)


class TurboCodec(AFF3CTCodec):
    """Turbo codec using DVB-RCS2 standard components.

    Turbo codes are parallel concatenated convolutional codes with
    iterative decoding. This implementation uses the DVB-RCS2 standard
    which achieves near-capacity performance.

    Note:
        Due to the DVB-RCS2 interleaver constraints, only specific K values
        are supported. Use TurboCodec.valid_k_values() to get the list.
    """

    # Valid K//2 sizes for DVB-RCS2 interleaver
    _VALID_HALF_K = [56, 64, 152, 204, 228, 340, 384, 400, 432, 440,
                    652, 680, 752, 864, 1192, 1332, 1504, 1752]

    @classmethod
    def valid_k_values(cls) -> list:
        """Return list of valid K values for DVB-RCS2 Turbo codec."""
        return [2 * half_k for half_k in cls._VALID_HALF_K]

    def __init__(self, k: int, max_iter: int = 8):
        """Initialize Turbo codec (DVB-RCS2).

        Args:
            k: Message length. Must be from the valid set (see valid_k_values()).
               Common values: 128, 456, 680, 768, 800, 864, 880, 1360, 1504, 1728
            max_iter: Maximum decoder iterations (default: 8)

        Note:
            Code rate is approximately 1/3. Due to DVB-RCS2 interleaver design,
            the first few decoded bits may have slightly higher error rates.

        Raises:
            ValueError: If k is not a valid DVB-RCS2 size
        """
        if k % 2 != 0:
            raise ValueError(f"Message length k={k} must be even for DVB-RCS2 turbo codes")

        if k // 2 not in self._VALID_HALF_K:
            valid_ks = self.valid_k_values()
            raise ValueError(
                f"Message length k={k} is not a valid DVB-RCS2 size. "
                f"K//2 must be one of: {self._VALID_HALF_K}. "
                f"Valid K values: {valid_ks}"
            )

        n = 3 * k  # Rate ~1/3
        super().__init__(k, n)
        self._max_iter = max_iter

        # Build constituent RSC encoders
        enc_n = self._aff3ct.module.encoder.Encoder_RSC_DB(k, 2 * k, standard='DVB-RCS2')
        enc_i = self._aff3ct.module.encoder.Encoder_RSC_DB(k, 2 * k, standard='DVB-RCS2')

        # Build interleaver
        itl_core = self._aff3ct.tools.interleaver_core.Interleaver_core_ARP_DVB_RCS2(k // 2)
        itl_bit = self._aff3ct.module.interleaver.Interleaver_int32(itl_core)
        itl_llr = self._aff3ct.module.interleaver.Interleaver_float(itl_core)

        # Build turbo encoder
        self._encoder = self._aff3ct.module.encoder.Encoder_turbo_DB(k, n, enc_n, enc_i, itl_bit)

        # Build turbo decoder
        trellis_n = enc_n.get_trellis()
        trellis_i = enc_i.get_trellis()
        dec_n = self._aff3ct.module.decoder.Decoder_RSC_DB_BCJR_DVB_RCS2(k, trellis_n)
        dec_i = self._aff3ct.module.decoder.Decoder_RSC_DB_BCJR_DVB_RCS2(k, trellis_i)
        self._decoder = self._aff3ct.module.decoder.Decoder_turbo_DB(
            k, n, max_iter, dec_n, dec_i, itl_llr
        )

        # Store all components to prevent garbage collection
        # (py_aff3ct uses internal references that require objects to stay alive)
        self._enc_n = enc_n
        self._enc_i = enc_i
        self._itl_core = itl_core
        self._itl_bit = itl_bit
        self._itl_llr = itl_llr
        self._trellis_n = trellis_n
        self._trellis_i = trellis_i
        self._dec_n = dec_n
        self._dec_i = dec_i

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encode message bits to Turbo codeword.

        Args:
            bits: 1D array of message bits (will be padded to multiple of K)

        Returns:
            Encoded codeword bits
        """
        bits = np.asarray(bits, dtype=np.int32).flatten()

        # Pad to multiple of K
        n_blocks = int(np.ceil(len(bits) / self._k))
        padded = np.zeros(n_blocks * self._k, dtype=np.int32)
        padded[:len(bits)] = bits

        # Encode each block
        encoded = []
        for i in range(n_blocks):
            block = padded[i * self._k : (i + 1) * self._k]
            # AFF3CT expects 2D array with shape (1, K)
            block_2d = block.reshape(1, -1)
            self._encoder["encode::U_K"].bind(block_2d)
            self._encoder["encode"].exec()
            encoded.append(self._encoder["encode::X_N"][:].flatten().copy())

        return np.concatenate(encoded).astype(np.uint8)

    def decode_soft(self, llrs: np.ndarray) -> np.ndarray:
        """Decode using log-likelihood ratios (soft-decision).

        Args:
            llrs: 1D array of LLRs (positive = more likely bit 0)

        Returns:
            Decoded message bits
        """
        llrs = np.asarray(llrs, dtype=np.float32).flatten()
        n_blocks = len(llrs) // self._n

        decoded = []
        for i in range(n_blocks):
            block = llrs[i * self._n : (i + 1) * self._n]
            # AFF3CT expects 2D array with shape (1, N)
            block_2d = block.reshape(1, -1)
            self._decoder["decode_siho::Y_N"].bind(block_2d)
            self._decoder["decode_siho"].exec()
            decoded.append(self._decoder["decode_siho::V_K"][:].flatten().copy())

        return np.concatenate(decoded).astype(np.uint8)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        """Decode using hard-decision bits."""
        bits = np.asarray(bits, dtype=np.float32).flatten()
        llrs = 1.0 - 2.0 * bits
        return self.decode_soft(llrs)

"""
Forward Error Correction (FEC) Codec Interface

Provides an extensible interface for FEC codecs used in metadata transmission.
Currently implements a passthrough codec; additional codecs (repetition,
convolutional, LDPC) can be added by subclassing FECCodec.
"""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class FECCodec(ABC):
    """Abstract base class for Forward Error Correction codecs.

    Subclasses must implement encode() and decode() methods, and define
    the code rate property. Advanced codecs can also implement soft-decision
    decoding via decode_soft().
    """

    @abstractmethod
    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Encode input bits with error correction.

        Args:
            bits: 1D numpy array of binary values (0 or 1)

        Returns:
            Encoded bit array (length = input_length / rate)
        """
        pass

    @abstractmethod
    def decode(self, bits: np.ndarray) -> np.ndarray:
        """Decode received bits and correct errors (hard-decision).

        Args:
            bits: 1D numpy array of (possibly corrupted) binary values

        Returns:
            Decoded bit array (original message bits)
        """
        pass

    @property
    @abstractmethod
    def rate(self) -> float:
        """Code rate (k/n where k=info bits, n=coded bits).

        Returns:
            Float between 0 and 1 (e.g., 0.5 for rate-1/2 code)
        """
        pass

    def supports_soft_decoding(self) -> bool:
        """Check if this codec supports soft-decision (LLR) decoding.

        Returns:
            True if decode_soft() is implemented, False otherwise.
        """
        return False

    def decode_soft(self, llrs: np.ndarray) -> np.ndarray:
        """Decode using log-likelihood ratios (soft-decision decoding).

        This method provides better error correction performance than
        hard-decision decode() by using channel reliability information.

        Args:
            llrs: 1D numpy array of log-likelihood ratios.
                  Convention: positive LLR = more likely bit 0,
                             negative LLR = more likely bit 1

        Returns:
            Decoded bit array (original message bits)

        Raises:
            NotImplementedError: If soft decoding is not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support soft-decision decoding. "
            "Use decode() for hard-decision decoding."
        )

    def encoded_length(self, input_length: int) -> int:
        """Calculate output length after encoding.

        Args:
            input_length: Number of input bits

        Returns:
            Number of output bits after encoding
        """
        return int(np.ceil(input_length / self.rate))

    def decoded_length(self, encoded_length: int) -> int:
        """Calculate original message length from encoded length.

        Args:
            encoded_length: Number of encoded bits

        Returns:
            Number of original message bits
        """
        return int(encoded_length * self.rate)


class PassthroughFEC(FECCodec):
    """No-op FEC codec that passes bits unchanged.

    Use this as a baseline or when FEC is disabled. The codec simply
    returns the input bits without modification.
    """

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Pass bits through unchanged."""
        return bits.copy()

    def decode(self, bits: np.ndarray) -> np.ndarray:
        """Pass bits through unchanged."""
        return bits.copy()

    @property
    def rate(self) -> float:
        """Rate 1.0 (no redundancy added)."""
        return 1.0


class RepetitionFEC(FECCodec):
    """Simple repetition code that repeats each bit N times.

    Decoding uses majority voting. Simple but effective for very
    short messages with high SNR requirements.

    Args:
        repetitions: Number of times to repeat each bit (must be odd for
                    unambiguous majority voting)
    """

    def __init__(self, repetitions: int = 3):
        if repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        if repetitions % 2 == 0:
            # Even repetitions can cause ties; warn but allow
            import warnings
            warnings.warn(
                f"Even repetition count ({repetitions}) may cause ambiguous "
                "decoding. Consider using an odd number.",
                UserWarning
            )
        self._repetitions = repetitions

    def encode(self, bits: np.ndarray) -> np.ndarray:
        """Repeat each bit N times."""
        bits = np.asarray(bits).flatten()
        return np.repeat(bits, self._repetitions)

    def decode(self, bits: np.ndarray) -> np.ndarray:
        """Decode using majority voting."""
        bits = np.asarray(bits).flatten()
        n = self._repetitions
        # Pad to multiple of n if necessary
        pad_len = (n - len(bits) % n) % n
        if pad_len > 0:
            bits = np.concatenate([bits, np.zeros(pad_len)])

        # Reshape and vote
        reshaped = bits.reshape(-1, n)
        # Majority vote: if sum > n/2, output 1; else 0
        decoded = (reshaped.sum(axis=1) > n / 2).astype(np.uint8)
        return decoded

    @property
    def rate(self) -> float:
        """Rate = 1/repetitions."""
        return 1.0 / self._repetitions


def get_fec_codec(name: str, **kwargs) -> FECCodec:
    """Factory function to create FEC codec by name.

    Args:
        name: Codec name. Basic codecs: 'none', 'passthrough', 'repetition'.
              Advanced codecs (requires py_aff3ct): 'ldpc', 'polar', 'turbo', 'rsc'.
        **kwargs: Additional arguments passed to codec constructor

    Returns:
        FECCodec instance

    Raises:
        ValueError: If codec name is not recognized
        ImportError: If advanced codec requested but py_aff3ct not available
    """
    # Basic codecs (always available)
    basic_codecs = {
        'none': PassthroughFEC,
        'passthrough': PassthroughFEC,
        'repetition': RepetitionFEC,
    }

    name_lower = name.lower()

    # Check basic codecs first
    if name_lower in basic_codecs:
        return basic_codecs[name_lower](**kwargs)

    # Advanced codecs require py_aff3ct (lazy import)
    advanced_codec_names = ['ldpc', 'polar', 'turbo', 'rsc', 'convolutional']

    if name_lower in advanced_codec_names:
        try:
            from python_replicate.aff3ct_codecs import (
                LDPCCodec, PolarCodec, TurboCodec, RSCCodec
            )
            aff3ct_codecs = {
                'ldpc': LDPCCodec,
                'polar': PolarCodec,
                'turbo': TurboCodec,
                'rsc': RSCCodec,
                'convolutional': RSCCodec,
            }
            return aff3ct_codecs[name_lower](**kwargs)
        except ImportError as e:
            raise ImportError(
                f"py_aff3ct is required for '{name}' codec but could not be imported. "
                f"Ensure py_aff3ct is built and available. Error: {e}"
            )

    # Unknown codec
    available = list(basic_codecs.keys()) + advanced_codec_names
    raise ValueError(f"Unknown FEC codec '{name}'. Available: {', '.join(available)}")

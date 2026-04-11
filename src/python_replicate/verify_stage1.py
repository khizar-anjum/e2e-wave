from __future__ import annotations

import pathlib

import torch
from scipy.io import loadmat

from .frame_preparation import FramePrepConfig, prepare_frame


def main() -> None:
    mat_path = pathlib.Path("matlab") / "qpsk_signal_OFDM.mat"
    ref = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    params = ref["params"]
    data_bits = torch.tensor(params.data_bits.astype(int), dtype=torch.int64)
    result = prepare_frame(FramePrepConfig(), data_bits=data_bits)

    ref_packet = torch.from_numpy(params.packet).to(torch.cdouble).view(-1)
    ref_rrc = torch.from_numpy(params.rrc).to(torch.float64)
    ref_sync = torch.from_numpy(params.sync_seq).to(torch.cdouble).view(-1)
    ref_x = torch.from_numpy(ref["x"]).to(torch.float64).view(-1)

    packet_rmse = torch.sqrt(
        torch.mean(torch.abs(result.packet_baseband - ref_packet) ** 2)
    )
    rrc_max_err = torch.max(torch.abs(result.rrc - ref_rrc))
    sync_rmse = torch.sqrt(
        torch.mean(torch.abs(result.sync_seq - ref_sync) ** 2)
    )
    passband_rmse = torch.sqrt(
        torch.mean((result.passband - ref_x) ** 2)
    )

    print("Stage 1 – Frame preparation")
    print(f" Packet RMSE     : {packet_rmse.item():.3e}")
    print(f" Passband RMSE   : {passband_rmse.item():.3e}")
    print(f" RRC max error   : {rrc_max_err.item():.3e}")
    print(f" Sync seq RMSE   : {sync_rmse.item():.3e}")


if __name__ == "__main__":
    main()


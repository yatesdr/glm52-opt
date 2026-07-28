# CN4 Intel ME update runbook

Date prepared: 2026-07-24

## Scope

Update the ASUS WS C422 SAGE/10G Intel Management Engine firmware from
`11.12.0.1622 H` to the ASUS-published `11.12.98.2655` image.

The firmware write and reboot were explicitly approved by Derek and
completed successfully on 2026-07-24. No executable flash script was
installed or scheduled.

## Completion record

```text
Pre-update firmware:  11.12.0.1622 H
Target image:         11.12.98.2655
FWUpdLcl result:      FW Update is completed successfully
Command exit code:    0
Post-reboot firmware: 11.12.98.2655 H
MEInfo result:        Normal / Valid / Complete / No Error
```

Evidence retained on CN4:

```text
/home/derek/me-update-staging-20260724/flash-transcript-20260724.log
  sha256 db479ce8115e5111bdad33e12052046e875f99a3cec863b7a35da2d975f64411

/home/derek/me-update-staging-20260724/post-update-meinfo-20260724.log
  sha256 6a1d62abcdb49c05ab1d6c1d72e252752ff73662dd464f5c2cf8e2f9656932c9
```

The update was run interactively over MEI with no `-Y`, `-ALLOWSV`,
`-PARTID`, or `-FORCERESET` option. The image completed firmware-side
verification before the write began. A controlled host reboot was then
performed.

Post-reboot hardware audit:

- all four RTX PRO 6000 GPUs enumerated at x16 with Gen3 capability;
- both NVMe devices enumerated and their expected filesystems mounted;
- all exposed PCIe AER counters were zero;
- no boot-log AER, MCE, NVRM Xid, MEI, NVMe reset, or NVMe timeout error;
- Samsung and Intel NVMe SMART both reported zero critical warnings and
  zero media errors;
- the vLLM test containers were deliberately left stopped.

Two unrelated observations remain:

- The Samsung EFI system partition was reported as not cleanly unmounted,
  consistent with CN4's earlier hard-reset history. It was not modified
  during this firmware task.
- The Samsung SMART log records 60 historical unsafe shutdowns and three
  counted error-log entries. All currently exposed error-log records are
  zero/successful, and the drive reports zero media errors.

## Staged payload on CN4

Directory:

```text
/home/derek/me-update-staging-20260724
```

Files:

```text
ASUS-MEUpdateTool-11.12.98.2655.zip
  sha256 144415ba81df36dbc64ab234a1e95ec20b68a39e0346c49065ed259681848020

ASUS-ME-11.12.98.2655.bin
  sha256 a3d8799c0a7971dde9c8058f5be1e0090ac888ae44ac503429c352c487901385

FWUpdLcl-LINUX64-11.8.92.4222
  sha256 2c8ac28fcd7eb8c445cca09effc6f427e6c01c14cfb3fdd1bfac360e8c93142b

MEInfo-LINUX64-11.8.92.4222
  sha256 ca85f8a087bd9b4b9a64b000dda3069b7e772f3334ed7d86d0044e56f310dfba
```

The firmware archive hash exactly matches ASUS's published hash. ASUS
bundles a Windows updater, not a Linux updater. The Linux utilities are
Intel CSME v11 OEM tools from the Win-Raid/Level1Techs system-tools
archive. They are older than the target image, so all compatibility checks
must fail closed.

## Completed read-only preflight

`FWUpdLcl -FWVER` results:

```text
Installed FW: 11.12.0.1622
ASUS image:   11.12.98.2655
```

`MEInfo -verbose` results:

```text
FW Version:                    11.12.0.1622 H
CurrentState:                  Normal
ManufacturingMode:             Disabled
FlashPartition:                Valid
InitComplete:                  Complete
BUPLoadState:                  Success
ErrorCode:                     No Error
ModeOfOperation:               Normal
ICC:                           Valid OEM data, ICC programmed
ME File System Corrupted:      No
FPF and ME Config Status:      Match
Local FWUpdate:                Enabled
BIOS Config Lock:              Enabled
Host Write Access to ME:       Disabled
```

`Local FWUpdate: Enabled` is the relevant supported update path.
`Host Write Access to ME: Disabled` blocks raw host SPI writes and is not
an obstacle to the firmware-managed FWUpdate path.

The final `Unable to clean up MEInfo before exiting` message was emitted
after successful reporting with exit code zero. The ME status itself
reported no error.

## Flash-window prerequisites

Do not begin until all of the following are true:

1. Derek has explicitly approved the flash.
2. CN4's vLLM/testing workload has been intentionally stopped.
3. No important filesystem writes, package operations, or remote jobs are
   active.
4. AC power is stable and no maintenance is occurring on the electrical
   circuit.
5. Someone can physically recover or power-cycle CN4 if it does not return
   remotely.
6. Re-run the SHA-256 checks and the read-only installed/image version
   checks immediately before flashing.

## Gated flash command

The following command was executed once under the approved maintenance
window. Do not repeat it:

```bash
cd /home/derek/me-update-staging-20260724
sudo ./FWUpdLcl-LINUX64-11.8.92.4222 \
  -F ASUS-ME-11.12.98.2655.bin
```

Run interactively without `-Y`, `-ALLOWSV`, `-PARTID`, or
`-FORCERESET`. Preserve the complete output.

Stop immediately without bypass flags if the utility reports any of:

- incompatible image, SKU, platform, partition, SVN, or VCN;
- unsupported firmware/tool version;
- disabled local firmware update;
- image authentication or signature failure;
- inability to communicate with the ME;
- any request to override or force compatibility.

Do not substitute `flashrom`, Intel FPT, raw SPI access, or Wine.

## Post-update verification

After a successful utility result:

1. Perform the reboot requested by the updater.
2. Verify `/sys/class/mei/mei0/fw_ver` reports `11.12.98.2655`.
3. Run `MEInfo -verbose` again and require Normal / No Error / Valid /
   Complete status.
4. Check the complete boot kernel log for MEI, PCIe AER, NVMe, MCE, Xid,
   watchdog, and firmware errors.
5. Verify all four GPUs, negotiated PCIe widths/speeds, NVMe devices, and
   network interfaces before restarting vLLM.
6. Resume the stability workload only after the hardware audit is clean.

## Interpretation

The update is justified by ASUS's BIOS prerequisite and the large firmware
revision gap. It may improve platform initialization, security, and ME
firmware behavior. It is not an AER logger upgrade and is not, by itself,
proof that the prior PCIe/reset symptoms were caused by ME firmware.

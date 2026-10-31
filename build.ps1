# Build the source and assemble a clean, directly copyable USB directory.
# The lower-level make-usbroot.ps1 deliberately emits flat extraction output;
# this wrapper stages it and copies only the deployment manifest into usbroot.
#
# Usage:
#   .\build.ps1                          # Turing (RTX 20 / GTX 16)
#   .\build.ps1 -Family 30               # Ampere (RTX 30)
#   .\build.ps1 -Family 40               # Ada (RTX 40)
#   .\build.ps1 -BuildOnly               # Build and test source only

param(
    [ValidateSet("ad102", "ga102", "tu10x", "rtx40", "rtx30", "rtx20", "40", "30", "20")]
    [string]$Family = "20",
    [string]$Out = "usbroot",
    [string]$Work = "inputs",
    [string]$ProvenBlob = "",
    [switch]$SkipGspCheck,
    [switch]$BuildOnly
)

$ErrorActionPreference = "Stop"
$root = [System.IO.Path]::GetFullPath($PSScriptRoot)
$assetsExe = Join-Path $root "target\release\assets.exe"
$bootEfi = Join-Path $root "target\x86_64-unknown-uefi\release\loader.efi"
$ssdiagExe = Join-Path $root "target\ssdiag\release\ssdiag.exe"
$ssdiagUsbName = "ssdiag-x86_64-windows.exe"
$gpuBenchmarkExe = Join-Path $root "target\gpubench\release\gpu-benchmark.exe"
$gpuBenchmarkUsbName = "gpu-benchmark-x86_64-windows.exe"
$maker = Join-Path $root "release\make-usbroot.ps1"
$stagePath = Join-Path $root "target\usbroot-stage"
$publishPath = Join-Path $root "target\usbroot-publish"

function Resolve-ProjectPath([string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $root $Path))
}

function Normalize-DirectoryPath([string]$Path) {
    $full = [System.IO.Path]::GetFullPath($Path)
    $volumeRoot = [System.IO.Path]::GetPathRoot($full)
    if ($full.Length -gt $volumeRoot.Length) {
        $full = $full.TrimEnd([char]'\', [char]'/')
    }
    return $full
}

function Test-PathWithin([string]$Path, [string]$Parent) {
    $childPath = Normalize-DirectoryPath $Path
    $parentPath = Normalize-DirectoryPath $Parent
    $prefix = $parentPath + [System.IO.Path]::DirectorySeparatorChar
    return $childPath.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)
}

$outPath = Resolve-ProjectPath $Out
$workPath = Resolve-ProjectPath $Work
if ($ProvenBlob) { $ProvenBlob = Resolve-ProjectPath $ProvenBlob }

$familyKey = switch ($Family.ToLowerInvariant()) {
    { $_ -in "ad102", "rtx40", "40" } { "ad102" }
    { $_ -in "ga102", "rtx30", "30" } { "ga102" }
    { $_ -in "tu10x", "rtx20", "20" } { "tu10x" }
}

$normalizedOut = Normalize-DirectoryPath $outPath
$volumeRoot = Normalize-DirectoryPath ([System.IO.Path]::GetPathRoot($outPath))
if ($normalizedOut -eq $volumeRoot -or $normalizedOut -eq $root) {
    throw "-Out must be a dedicated output directory, not a drive root or the project root: $outPath"
}
foreach ($protectedPath in @($workPath, $stagePath, $publishPath)) {
    $normalizedProtected = Normalize-DirectoryPath $protectedPath
    if ($normalizedOut -eq $normalizedProtected -or
        (Test-PathWithin $normalizedProtected $normalizedOut) -or
        (Test-PathWithin $normalizedOut $normalizedProtected)) {
        throw "-Out must not overlap the input cache or internal staging directories: $outPath"
    }
}

if (-not (Get-Command cargo.exe -ErrorAction SilentlyContinue)) {
    throw "cargo.exe not found in PATH. Install Rust with the Windows MSVC toolchain."
}
if (-not (Test-Path $maker)) {
    throw "USB root generation script not found: $maker"
}

Push-Location $root
try {
    $rustup = Get-Command rustup.exe -ErrorAction SilentlyContinue
    if ($rustup) {
        $installedTargets = @(& $rustup.Source target list --installed)
        if ($LASTEXITCODE -ne 0) { throw "Unable to query installed Rust targets." }
        if ($installedTargets -notcontains "x86_64-unknown-uefi") {
            Write-Host "[toolchain] Installing required Rust target x86_64-unknown-uefi ..."
            & $rustup.Source target add x86_64-unknown-uefi
            if ($LASTEXITCODE -ne 0) { throw "Failed to install Rust target x86_64-unknown-uefi." }
        }
    } else {
        Write-Warning "rustup.exe not found; assuming x86_64-unknown-uefi is already installed."
    }

    Write-Host "[1/6] Building Windows assets and simulator (release) ..."
    & cargo build --release --locked
    if ($LASTEXITCODE -ne 0) { throw "Host source build failed (cargo exit code $LASTEXITCODE)." }
    if (-not (Test-Path $assetsExe)) { throw "Missing compiled assets tool: $assetsExe" }

    Write-Host "[2/6] Building UEFI transaction loader (release, x86_64-unknown-uefi) ..."
    & cargo efi-tx --locked
    if ($LASTEXITCODE -ne 0) { throw "UEFI loader source build failed (cargo exit code $LASTEXITCODE)." }
    if (-not (Test-Path $bootEfi)) { throw "Missing compiled EFI loader: $bootEfi" }

    Write-Host "[3/6] Running host-side workspace tests ..."
    & cargo xt --locked
    if ($LASTEXITCODE -ne 0) { throw "Host-side tests failed (cargo exit code $LASTEXITCODE)." }

    $rustcInfo = @(& rustc.exe -vV)
    if ($LASTEXITCODE -ne 0) { throw "Unable to query Rust compiler host target." }
    $rustHost = ($rustcInfo | Where-Object { $_ -like "host: *" } | Select-Object -First 1) -replace '^host:\s*', ''
    if ($rustHost -notmatch '^x86_64-pc-windows-') {
        throw "The USB Windows tools require an x86_64 Windows Rust host; detected '$rustHost'."
    }

    Write-Host "[4/6] Building standalone ssdiag Windows diagnostic tool ($rustHost) ..."
    & cargo build --release --locked --manifest-path tools/ssdiag/Cargo.toml --target-dir target/ssdiag
    if ($LASTEXITCODE -ne 0) { throw "ssdiag build failed (cargo exit code $LASTEXITCODE)." }
    if (-not (Test-Path -LiteralPath $ssdiagExe -PathType Leaf)) { throw "Missing compiled ssdiag executable: $ssdiagExe" }
    if ((Get-Item -LiteralPath $ssdiagExe).Length -le 0) { throw "Compiled ssdiag executable is empty: $ssdiagExe" }

    Write-Host "[5/6] Testing and building standalone Windows GPU benchmark ..."
    & cargo test --locked --manifest-path tools/gpubench/Cargo.toml --target-dir target/gpubench
    if ($LASTEXITCODE -ne 0) { throw "GPU benchmark tests failed (cargo exit code $LASTEXITCODE)." }
    & cargo build --release --locked --manifest-path tools/gpubench/Cargo.toml --target-dir target/gpubench
    if ($LASTEXITCODE -ne 0) { throw "GPU benchmark build failed (cargo exit code $LASTEXITCODE)." }
    if (-not (Test-Path -LiteralPath $gpuBenchmarkExe -PathType Leaf)) { throw "Missing compiled GPU benchmark: $gpuBenchmarkExe" }
    if ((Get-Item -LiteralPath $gpuBenchmarkExe).Length -le 0) { throw "Compiled GPU benchmark is empty: $gpuBenchmarkExe" }

    if ($BuildOnly) {
        Write-Host "[ok] Host tools, UEFI loader, ssdiag, GPU benchmark and host-side tests built successfully (USB assembly skipped)."
        return
    }

    # Never let the flat maker output leak into the user-facing usbroot.
    # Both staging directories live under target/ (gitignored) and are
    # recreated on each run so no stale family assets can be mixed in.
    foreach ($directory in @($stagePath, $publishPath)) {
        if (Test-Path $directory) { Remove-Item -LiteralPath $directory -Recurse -Force }
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }

    Write-Host "[6/6] Generating $familyKey assets and assembling USB layout ..."
    $makerArgs = @{
        Family = $Family
        AssetsExe = $assetsExe
        BootEfi = $bootEfi
        Out = $stagePath
        Work = $workPath
        SkipGspCheck = $SkipGspCheck
    }
    if ($ProvenBlob) { $makerArgs.ProvenBlob = $ProvenBlob }
    & $maker @makerArgs
    if (-not $?) { throw "USB asset generation failed." }

    # AGENTS.deploy.md's exact per-family USB manifest. Do not copy every
    # .bin from the maker directory: it intentionally contains intermediates.
    $rootFiles = @(
        "booter_load_$familyKey.bin",
        "gsp_boot_desc_$familyKey.bin",
        "gsp_boot_$familyKey.bin",
        "fwimage_$familyKey.bin",
        "fwsignature_$familyKey.bin",
        "v67-$familyKey-four-plm.bin",
        $ssdiagUsbName,
        $gpuBenchmarkUsbName
    )
    if ($familyKey -in @("ad102", "ga102")) {
        $rootFiles += "scrubber_ad10x.bin"
    } else {
        $rootFiles += "sec2_bl_tu10x.bin"
        $rootFiles += "gsp-gate.txt"
    }

    $makerFiles = @("BOOTX64.EFI") + @($rootFiles | Where-Object { $_ -notin @($ssdiagUsbName, $gpuBenchmarkUsbName) })
    foreach ($file in $makerFiles) {
        $source = Join-Path $stagePath $file
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Required deployment file is missing from maker output: $source"
        }
        if ((Get-Item -LiteralPath $source).Length -le 0) {
            throw "Required deployment file is empty: $source"
        }
    }

    # Build the exact destination tree in a separate publish directory first.
    $efiBootDirectory = Join-Path $publishPath "EFI\BOOT"
    New-Item -ItemType Directory -Force -Path $efiBootDirectory | Out-Null
    Copy-Item -LiteralPath (Join-Path $stagePath "BOOTX64.EFI") -Destination (Join-Path $efiBootDirectory "BOOTX64.EFI") -Force
    foreach ($file in $rootFiles) {
        if ($file -eq $ssdiagUsbName) {
            Copy-Item -LiteralPath $ssdiagExe -Destination (Join-Path $publishPath $file) -Force
        } elseif ($file -eq $gpuBenchmarkUsbName) {
            Copy-Item -LiteralPath $gpuBenchmarkExe -Destination (Join-Path $publishPath $file) -Force
        } else {
            Copy-Item -LiteralPath (Join-Path $stagePath $file) -Destination (Join-Path $publishPath $file) -Force
        }
    }

    # Retain only documented, explicitly configured optional root markers.
    # The unattended-execution marker is intentionally never carried forward.
    $optionalMarkers = @("tx-require-d0.txt", "tx-select-bdf.txt", "chain-target.txt", "no-reconnect.txt")
    if ($familyKey -eq "tu10x") { $optionalMarkers += "tu10x-fb-gb.txt" }
    $preservedMarkers = @()
    foreach ($marker in $optionalMarkers) {
        $existingMarker = Join-Path $outPath $marker
        if (Test-Path -LiteralPath $existingMarker -PathType Leaf) {
            if ($marker -eq "tx-select-bdf.txt") {
                $bdf = (Get-Content -LiteralPath $existingMarker -Raw).Trim()
                if ($bdf -notmatch '^(?i)([0-9a-f]{2}:[0-9a-f]{2}\.[0-7]|[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])$') {
                    throw "Invalid tx-select-bdf.txt in existing output: '$bdf'. Expected bb:dd.f or dddd:bb:dd.f."
                }
            }
            Copy-Item -LiteralPath $existingMarker -Destination (Join-Path $publishPath $marker) -Force
            $preservedMarkers += $marker
            Write-Host "[keep] optional USB marker $marker"
        }
    }

    # AGENTS.deploy.md requires an existing auto-run marker to be backed up
    # outside usbroot, then omitted unless unattended execution is explicitly
    # requested. This build script has no unattended mode.
    $armedMarker = Join-Path $outPath "i_know_what_i_am_doing.txt"
    if (Test-Path -LiteralPath $armedMarker -PathType Leaf) {
        $backupDirectory = Join-Path $workPath "usbroot-marker-backups"
        New-Item -ItemType Directory -Force -Path $backupDirectory | Out-Null
        $backupName = "i_know_what_i_am_doing-{0}.txt" -f (Get-Date -Format "yyyyMMdd-HHmmss-fff")
        $backupPath = Join-Path $backupDirectory $backupName
        Copy-Item -LiteralPath $armedMarker -Destination $backupPath -Force
        Write-Warning "Backed up unattended-execution marker outside usbroot: $backupPath; it will not be included in the new output."
    }

    # Keep the previous run's evidence before replacing a prepared output tree.
    $previousLog = Join-Path $outPath "loader.log"
    if (Test-Path -LiteralPath $previousLog -PathType Leaf) {
        $logArchiveDirectory = Join-Path $workPath "usbroot-logs"
        New-Item -ItemType Directory -Force -Path $logArchiveDirectory | Out-Null
        $logArchiveName = "loader-{0}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss-fff")
        $logArchivePath = Join-Path $logArchiveDirectory $logArchiveName
        Copy-Item -LiteralPath $previousLog -Destination $logArchivePath -Force
        Write-Warning "Archived previous loader.log outside usbroot: $logArchivePath"
    }

    # Publish only after every required source file has passed validation.
    if (Test-Path -LiteralPath $outPath) {
        if (-not (Test-Path -LiteralPath $outPath -PathType Container)) {
            throw "-Out exists but is not a directory: $outPath"
        }
        Get-ChildItem -LiteralPath $outPath -Force | Remove-Item -Recurse -Force
    } else {
        New-Item -ItemType Directory -Force -Path $outPath | Out-Null
    }
    Get-ChildItem -LiteralPath $publishPath -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $outPath -Recurse -Force
    }

    # Verify the published tree against the exact USB manifest. This catches
    # both missing files and accidental leakage of maker intermediates.
    $expectedRelativePaths = @("EFI\BOOT\BOOTX64.EFI") + $rootFiles + $preservedMarkers
    $actualRelativePaths = @(Get-ChildItem -LiteralPath $outPath -File -Recurse | ForEach-Object {
        $_.FullName.Substring($outPath.Length).TrimStart([char]'\', [char]'/')
    })
    $missingPaths = @($expectedRelativePaths | Where-Object { $_ -notin $actualRelativePaths })
    $unexpectedPaths = @($actualRelativePaths | Where-Object { $_ -notin $expectedRelativePaths })
    if ($missingPaths.Count -gt 0 -or $unexpectedPaths.Count -gt 0) {
        throw "Published USB tree does not match the manifest. Missing: $($missingPaths -join ', '); unexpected: $($unexpectedPaths -join ', ')"
    }

    # The final usbroot is now self-contained; discard the flat maker output
    # and the temporary publish tree so only the deployment tree is presented.
    Remove-Item -LiteralPath $stagePath, $publishPath -Recurse -Force

    Write-Host ""
    Write-Host "[ok] USB directory is ready: $outPath"
    Write-Host "Copy the CONTENTS of this directory to the root of a FAT32 USB stick."
    Write-Host ""
    Write-Host "EFI/BOOT/BOOTX64.EFI"
    foreach ($file in $rootFiles) {
        Write-Host $file
    }
    foreach ($marker in $preservedMarkers) { Write-Host $marker }
    Write-Host "[ok] Intermediate extraction files were excluded."
} finally {
    Pop-Location
}

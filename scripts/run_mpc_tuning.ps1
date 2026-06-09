param(
    [string[]]$Roots = @("Desktop", "Downloads"),
    [string[]]$Groups = @(),
    [string]$Python = "python",
    [int]$MaxSteps = 0,
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"

$DesktopRoot = "C:\Users\16321\Desktop\demo_docs_release_20260509"
$DownloadsRoot = "C:\Users\16321\Downloads\demo_docs_release_20260509"

$RootMap = @{
    Desktop = $DesktopRoot
    Downloads = $DownloadsRoot
}

$PureNet = @{
    Desktop = 52589.37
    Downloads = 247482.22
}

$Matrix = @(
    [ordered]@{ Id = "A0"; Weight = "0.10"; Cap = "100"; EffGap = "18"; NetGap = "180"; Ratio = "0.25"; PointVisit = "1.0" },
    [ordered]@{ Id = "A1"; Weight = "0.06"; Cap = "60"; EffGap = "12"; NetGap = "120"; Ratio = "0.15"; PointVisit = "1.0" },
    [ordered]@{ Id = "A2"; Weight = "0.08"; Cap = "80"; EffGap = "12"; NetGap = "150"; Ratio = "0.20"; PointVisit = "1.0" },
    [ordered]@{ Id = "A3"; Weight = "0.08"; Cap = "80"; EffGap = "18"; NetGap = "150"; Ratio = "0.20"; PointVisit = "1.0" },
    [ordered]@{ Id = "A4"; Weight = "0.10"; Cap = "80"; EffGap = "12"; NetGap = "150"; Ratio = "0.20"; PointVisit = "1.0" },
    [ordered]@{ Id = "A5"; Weight = "0.10"; Cap = "100"; EffGap = "12"; NetGap = "120"; Ratio = "0.15"; PointVisit = "1.0" },
    [ordered]@{ Id = "B1"; Weight = "0.10"; Cap = "100"; EffGap = "18"; NetGap = "180"; Ratio = "0.25"; PointVisit = "0" },
    [ordered]@{ Id = "B2"; Weight = "0.10"; Cap = "100"; EffGap = "18"; NetGap = "180"; Ratio = "0.25"; PointVisit = "0.35" },
    [ordered]@{ Id = "B3"; Weight = "0.08"; Cap = "80"; EffGap = "18"; NetGap = "150"; Ratio = "0.20"; PointVisit = "0" },
    [ordered]@{ Id = "B4"; Weight = "0.08"; Cap = "80"; EffGap = "18"; NetGap = "150"; Ratio = "0.20"; PointVisit = "0.35" },
    [ordered]@{ Id = "C1"; Weight = "0.12"; Cap = "120"; EffGap = "18"; NetGap = "180"; Ratio = "0.25"; PointVisit = "0" },
    [ordered]@{ Id = "C2"; Weight = "0.12"; Cap = "100"; EffGap = "12"; NetGap = "150"; Ratio = "0.20"; PointVisit = "0" }
)

function Resolve-RunGroups {
    param([array]$Matrix, [string[]]$GroupIds)
    $normalizedIds = @()
    foreach ($rawId in $GroupIds) {
        foreach ($part in ([string]$rawId -split ",")) {
            if (-not [string]::IsNullOrWhiteSpace($part)) {
                $normalizedIds += $part.Trim()
            }
        }
    }
    if ($normalizedIds.Count -eq 0) {
        return $Matrix
    }
    $wanted = @{}
    foreach ($id in $normalizedIds) {
        if ([string]::IsNullOrWhiteSpace($id)) {
            continue
        }
        $wanted[$id.Trim().ToUpperInvariant()] = $true
    }
    return @($Matrix | Where-Object { $wanted.ContainsKey([string]$_.Id) })
}

function Resolve-RootNames {
    param([string[]]$RequestedRoots)
    $normalized = @()
    foreach ($rawRoot in $RequestedRoots) {
        foreach ($part in ([string]$rawRoot -split ",")) {
            if (-not [string]::IsNullOrWhiteSpace($part)) {
                $normalized += $part.Trim()
            }
        }
    }
    return $normalized
}

function Count-AcceptedFalse {
    param([object]$RunSummary)
    $count = 0
    if ($null -eq $RunSummary -or $null -eq $RunSummary.driver_result_files) {
        return $count
    }
    foreach ($property in $RunSummary.driver_result_files.PSObject.Properties) {
        $path = [string]$property.Value
        if (-not (Test-Path -LiteralPath $path)) {
            continue
        }
        foreach ($line in Get-Content -LiteralPath $path -Encoding UTF8) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }
            try {
                $record = $line | ConvertFrom-Json
            } catch {
                continue
            }
            if ($null -ne $record.result -and $record.result.accepted -eq $false) {
                $count += 1
            }
        }
    }
    return $count
}

function Count-AcceptedFalseInDir {
    param([string]$RunDir)
    $count = 0
    foreach ($path in Get-ChildItem -LiteralPath $RunDir -Filter "actions_*.jsonl" -File) {
        foreach ($line in Get-Content -LiteralPath $path.FullName -Encoding UTF8) {
            if ([string]::IsNullOrWhiteSpace($line)) {
                continue
            }
            try {
                $record = $line | ConvertFrom-Json
            } catch {
                continue
            }
            if ($null -ne $record.result -and $record.result.accepted -eq $false) {
                $count += 1
            }
        }
    }
    return $count
}

function Quote-Arg {
    param([string]$Value)
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Invoke-IncomeCalculation {
    param([string]$RootPath, [string]$RunDir)
    $calcScript = Join-Path $RootPath "demo\calc_monthly_income.py"
    $projectRoot = Join-Path $RootPath "demo"
    if (-not (Test-Path -LiteralPath $calcScript)) {
        throw "Missing income calculator: $calcScript"
    }

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Python
    $psi.WorkingDirectory = $RootPath
    $psi.Arguments = @(
        "-B",
        (Quote-Arg $calcScript),
        "--project-root",
        (Quote-Arg $projectRoot),
        "--results-dir",
        (Quote-Arg $RunDir)
    ) -join " "
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $process = [System.Diagnostics.Process]::Start($psi)
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()
    $stdoutTask.GetAwaiter().GetResult() | Out-File -LiteralPath (Join-Path $RunDir "income_stdout.txt") -Encoding UTF8
    $stderrTask.GetAwaiter().GetResult() | Out-File -LiteralPath (Join-Path $RunDir "income_stderr.txt") -Encoding UTF8
    if ($process.ExitCode -ne 0) {
        throw "income calculation failed exit_code=$($process.ExitCode) run_dir=$RunDir"
    }
}

function Copy-RunArtifacts {
    param([string]$RootPath, [object]$RunSummary, [string]$RunDir)
    $resultsDir = Join-Path $RootPath "demo\results"
    foreach ($name in @("monthly_income_202603.json", "run_summary_202603.json")) {
        $path = Join-Path $resultsDir $name
        if (Test-Path -LiteralPath $path) {
            Copy-Item -LiteralPath $path -Destination $RunDir -Force
        }
    }
    if ($null -eq $RunSummary -or $null -eq $RunSummary.driver_result_files) {
        return
    }
    foreach ($property in $RunSummary.driver_result_files.PSObject.Properties) {
        $path = [string]$property.Value
        if (Test-Path -LiteralPath $path) {
            Copy-Item -LiteralPath $path -Destination $RunDir -Force
        }
    }
}

function Driver-NetSummary {
    param([object]$Monthly)
    if ($null -eq $Monthly -or $null -eq $Monthly.drivers) {
        return ""
    }
    $parts = @()
    foreach ($driver in $Monthly.drivers) {
        $parts += ("{0}={1}" -f $driver.driver_id, ([double]$driver.income.net_income).ToString("0.00"))
    }
    return ($parts -join ";")
}

function Start-MpcRun {
    param(
        [string]$RootName,
        [string]$RootPath,
        [object]$Group,
        [string]$RunDir
    )

    $psi = [System.Diagnostics.ProcessStartInfo]::new()
    $psi.FileName = $Python
    $psi.WorkingDirectory = $RootPath
    $arguments = @("-B", '".\demo\server\main.py"')
    if ($MaxSteps -gt 0) {
        $arguments += @("--max-steps", [string]$MaxSteps)
    }
    $psi.Arguments = ($arguments -join " ")
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.Environment["MPC_TERMINAL_VALUE_WEIGHT"] = [string]$Group.Weight
    $psi.Environment["MPC_TERMINAL_VALUE_CAP"] = [string]$Group.Cap
    $psi.Environment["MPC_MAX_EFFICIENCY_GAP_PER_HOUR"] = [string]$Group.EffGap
    $psi.Environment["MPC_MAX_NET_GAP"] = [string]$Group.NetGap
    $psi.Environment["MPC_MAX_NET_GAP_RATIO"] = [string]$Group.Ratio
    $psi.Environment["MPC_RULE_RISK_PROFILE"] = ('{"point_visit":' + [string]$Group.PointVisit + '}')

    $process = [System.Diagnostics.Process]::Start($psi)
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $started = Get-Date
    $process.WaitForExit()
    $ended = Get-Date

    $stdout = $stdoutTask.GetAwaiter().GetResult()
    $stderr = $stderrTask.GetAwaiter().GetResult()
    $stdout | Out-File -LiteralPath (Join-Path $RunDir "stdout.txt") -Encoding UTF8
    $stderr | Out-File -LiteralPath (Join-Path $RunDir "stderr.txt") -Encoding UTF8

    $runSummaryPath = Join-Path $RootPath "demo\results\run_summary_202603.json"
    $monthly = $null
    $runSummary = $null
    if (Test-Path -LiteralPath $runSummaryPath) {
        $runSummary = Get-Content -LiteralPath $runSummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }
    Copy-RunArtifacts -RootPath $RootPath -RunSummary $runSummary -RunDir $RunDir
    Invoke-IncomeCalculation -RootPath $RootPath -RunDir $RunDir

    $monthlyPath = Join-Path $RunDir "monthly_income_202603.json"
    if (Test-Path -LiteralPath $monthlyPath) {
        $monthly = Get-Content -LiteralPath $monthlyPath -Raw -Encoding UTF8 | ConvertFrom-Json
    }

    $net = $null
    $penalty = $null
    $tokens = $null
    $failed = $null
    $simulateSeconds = $null
    if ($null -ne $monthly -and $null -ne $monthly.summary) {
        $net = [double]$monthly.summary.total_net_income_all_drivers
        $penalty = [double]$monthly.summary.total_preference_penalty
        $tokens = [int]$monthly.summary.total_token_usage.total_tokens
        $failed = [int]$monthly.summary.failed_driver_count
        $simulateSeconds = [double]$monthly.simulate_time_seconds
    }
    $acceptedFalse = Count-AcceptedFalseInDir -RunDir $RunDir
    $delta = $null
    if ($null -ne $net -and $PureNet.ContainsKey($RootName)) {
        $delta = [Math]::Round($net - [double]$PureNet[$RootName], 2)
    }

    return [pscustomobject]@{
        group_id = $Group.Id
        dataset = $RootName
        exit_code = $process.ExitCode
        total_net = $net
        delta_vs_pure = $delta
        preference_penalty = $penalty
        accepted_false = $acceptedFalse
        failed_driver_count = $failed
        total_tokens = $tokens
        simulate_time_seconds = $simulateSeconds
        wall_seconds = [Math]::Round(($ended - $started).TotalSeconds, 2)
        driver_net = Driver-NetSummary -Monthly $monthly
        weight = $Group.Weight
        cap = $Group.Cap
        eff_gap = $Group.EffGap
        net_gap = $Group.NetGap
        ratio = $Group.Ratio
        point_visit = $Group.PointVisit
        output_dir = $RunDir
    }
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputRoot = Join-Path $DesktopRoot "demo\results\mpc_tuning_runs\$stamp"
}
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$RunGroups = Resolve-RunGroups -Matrix $Matrix -GroupIds $Groups
$RunRoots = Resolve-RootNames -RequestedRoots $Roots
if ($RunGroups.Count -eq 0) {
    throw "No matching groups. Valid group ids: $($Matrix.Id -join ', ')"
}
if ($RunRoots.Count -eq 0) {
    throw "No roots selected. Use Desktop, Downloads, or both."
}

$Results = @()
$SummaryCsv = Join-Path $OutputRoot "summary.csv"
$SummaryJson = Join-Path $OutputRoot "summary.json"

foreach ($group in $RunGroups) {
    foreach ($rootName in $RunRoots) {
        if (-not $RootMap.ContainsKey($rootName)) {
            throw "Unknown root '$rootName'. Use Desktop or Downloads."
        }
        $rootPath = $RootMap[$rootName]
        if (-not (Test-Path -LiteralPath (Join-Path $rootPath "demo\server\main.py"))) {
            throw "Missing demo\server\main.py under $rootPath"
        }
        $runDir = Join-Path $OutputRoot ("{0}_{1}" -f $group.Id, $rootName)
        New-Item -ItemType Directory -Path $runDir -Force | Out-Null
        $configPath = Join-Path $runDir "mpc_config.json"
        ($group | ConvertTo-Json -Depth 4) | Out-File -LiteralPath $configPath -Encoding UTF8

        Write-Host ("[{0}] running {1} on {2} ..." -f (Get-Date -Format "HH:mm:ss"), $group.Id, $rootName)
        $result = Start-MpcRun -RootName $rootName -RootPath $rootPath -Group $group -RunDir $runDir
        $Results += $result
        $Results | Export-Csv -LiteralPath $SummaryCsv -NoTypeInformation -Encoding UTF8
        $Results | ConvertTo-Json -Depth 5 | Out-File -LiteralPath $SummaryJson -Encoding UTF8
        Write-Host ("[{0}] done {1}/{2}: net={3} penalty={4} accepted_false={5} failed={6} exit={7}" -f `
            (Get-Date -Format "HH:mm:ss"), $group.Id, $rootName, $result.total_net, $result.preference_penalty, `
            $result.accepted_false, $result.failed_driver_count, $result.exit_code)
    }
}

Write-Host ""
Write-Host "Summary: $SummaryCsv"
Write-Host "Artifacts: $OutputRoot"



$inkpathLibPath = "C:\Program Files\Xournal++\share\xournalpp\plugins\ImageTranscription"

$userPath = [Environment]::GetEnvironmentVariable("PATH", [EnvironmentVariableTarget]::User)

if (-not ($userPath -split ";" | ForEach-Object { $_.Trim() } | Where-Object { $_ -eq $inkpathLibPath })) {
    $newPath = $userPath + ";" + $inkpathLibPath
    [Environment]::SetEnvironmentVariable("PATH", $newPath, [EnvironmentVariableTarget]::User)
    Write-Host "Configured Path."
} else {
    Write-Host "Path already updated."
}

Write-Host "Press any key to continue..."
[void][System.Console]::ReadKey($true)

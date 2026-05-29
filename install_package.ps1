param (
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$PackageName
)

$PythonExecutable = "C:\Programs\blender-5.2.0-alpha\5.2\python\bin\python.exe"
$TargetDir = "C:\git\blender\5.2\scripts\modules"

# Ensure target directory exists
if (-not (Test-Path $TargetDir)) {
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
}

Write-Host "Installing package '$PackageName' to '$TargetDir' using Blender Python..." -ForegroundColor Green

& $PythonExecutable -m pip install $PackageName --target $TargetDir --upgrade

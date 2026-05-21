$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$pythonCmd = Get-Command py -ErrorAction SilentlyContinue
if ($null -eq $pythonCmd) {
  $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
}

if ($null -eq $pythonCmd) {
  $fallbackPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"
  if (Test-Path $fallbackPython) {
    $pythonCmd = @{ Source = $fallbackPython }
  }
}

if ($null -eq $pythonCmd) {
  throw "No se encontro 'py' ni 'python' en el sistema."
}

$args = @(
  "-m", "PyInstaller",
  "--noconfirm",
  "--clean",
  "--onedir",
  "--windowed",
  "--name", "DetectorArroz",
  "--add-data", "rice_app\templates;rice_app\templates",
  "--add-data", "rice_app\static;rice_app\static"
)

$optionalFiles = @(
  "modelo_arroz_detect.pt",
  "modelo_arroz.pt",
  "modelo_cafe_detect.pt",
  "modelo_cafe.pt",
  "yolov8n.pt"
)

foreach ($file in $optionalFiles) {
  if (Test-Path $file) {
    $args += @("--add-data", "${file};.")
  }
}

$args += "web_python.py"

& $pythonCmd.Source @args

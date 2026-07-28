#Requires -Version 5.1
<#
Single entrypoint: builds/starts every container (postgres, redis, api,
all celery workers, beat, frontend, nginx) and opens the one public URL
(http://localhost:5000) once nginx is actually answering requests.
#>

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot
Set-Location $RepoRoot

$Url = 'http://localhost:5000'

function Test-DockerRunning {
    try {
        docker info *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

Write-Host "==> Checking Docker Desktop..." -ForegroundColor Cyan
if (-not (Test-DockerRunning)) {
    Write-Host "    Docker isn't responding, trying to start Docker Desktop..." -ForegroundColor Yellow
    $dockerExeCandidates = @(
        "$env:LOCALAPPDATA\Programs\DockerDesktop\Docker Desktop.exe",
        "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    )
    $dockerExe = $dockerExeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($dockerExe) {
        Start-Process $dockerExe
    } else {
        Write-Host "    Couldn't find Docker Desktop.exe automatically - start it manually." -ForegroundColor Yellow
    }
    $waited = 0
    while (-not (Test-DockerRunning)) {
        if ($waited -ge 180) {
            Write-Host "Docker didn't come up after 180s. Start Docker Desktop manually and re-run this script." -ForegroundColor Red
            exit 1
        }
        Start-Sleep -Seconds 3
        $waited += 3
        Write-Host "    ...still waiting for Docker ($waited s)"
    }
}
Write-Host "    Docker is up." -ForegroundColor Green

function Test-OllamaRunning {
    try {
        Invoke-WebRequest -Uri 'http://localhost:11434' -UseBasicParsing -TimeoutSec 2 | Out-Null
        return $true
    } catch {
        return $false
    }
}

Write-Host "==> Checking Ollama..." -ForegroundColor Cyan
if (-not (Test-OllamaRunning)) {
    Write-Host "    Ollama isn't responding, trying to start it..." -ForegroundColor Yellow
    $ollamaExeCandidates = @(
        "$env:LOCALAPPDATA\Programs\Ollama\ollama app.exe",
        "$env:ProgramFiles\Ollama\ollama app.exe"
    )
    $ollamaExe = $ollamaExeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($ollamaExe) {
        Start-Process $ollamaExe
    } else {
        Write-Host "    Couldn't find Ollama automatically - install it from https://ollama.com/download or start it manually." -ForegroundColor Yellow
    }
    $waited = 0
    while (-not (Test-OllamaRunning)) {
        if ($waited -ge 60) {
            Write-Host "    Ollama didn't come up after 60s - continuing anyway, but resume parsing / AI features will fail until it's running." -ForegroundColor Yellow
            break
        }
        Start-Sleep -Seconds 2
        $waited += 2
        Write-Host "    ...still waiting for Ollama ($waited s)"
    }
}
if (Test-OllamaRunning) {
    Write-Host "    Ollama is up." -ForegroundColor Green
}

Write-Host "==> Building and starting all services (postgres, redis, api, workers, beat, frontend, nginx)..." -ForegroundColor Cyan
docker compose up --build -d
if ($LASTEXITCODE -ne 0) {
    Write-Host "docker compose up failed - see output above." -ForegroundColor Red
    exit 1
}

Write-Host "==> Waiting for $Url to respond..." -ForegroundColor Cyan
$ready = $false
$waited = 0
$timeout = 180
$nginxRestarted = $false
while (-not $ready -and $waited -lt $timeout) {
    $statusCode = 0
    try {
        $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 3
        $statusCode = $resp.StatusCode
    } catch {
        if ($_.Exception.Response) {
            $statusCode = [int]$_.Exception.Response.StatusCode
        }
    }
    if ($statusCode -ge 200 -and $statusCode -lt 500) {
        $ready = $true
    } elseif ($statusCode -eq 502 -and $waited -ge 20 -and -not $nginxRestarted) {
        # nginx resolves the frontend/api container IPs once at worker
        # startup; if those containers were recreated afterward (fresh
        # build, IP changed), nginx keeps proxying to the stale address
        # until restarted so it re-resolves DNS.
        Write-Host "    nginx is up but proxying to a stale container IP, restarting it..." -ForegroundColor Yellow
        docker compose restart nginx | Out-Null
        $nginxRestarted = $true
    }
    if (-not $ready) {
        Start-Sleep -Seconds 2
        $waited += 2
    }
}

Write-Host ""
docker compose ps
Write-Host ""

if ($ready) {
    Write-Host "==> Ready! Opening $Url" -ForegroundColor Green
    Start-Process $Url
} else {
    Write-Host "==> Containers are starting but $Url isn't answering yet after ${timeout}s." -ForegroundColor Yellow
    Write-Host "    Check logs with: docker compose logs -f" -ForegroundColor Yellow
    Start-Process $Url
}

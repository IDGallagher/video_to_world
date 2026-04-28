$repo = (Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$drive = $repo.Substring(0, 1).ToLowerInvariant()
$repoTail = $repo.Substring(2) -replace "\\", "/"
$repoWsl = "/mnt/$drive$repoTail"
Set-Location $repo
wsl.exe -e bash -lc "cd '$repoWsl' && /home/idgal/miniforge3/envs/video_to_world/bin/python -u '$repoWsl/tmp_gradio_server.py'"

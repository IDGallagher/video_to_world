$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoWsl = (wsl.exe wslpath -a $repo).Trim()
Set-Location $repo
wsl.exe -e bash -lc "cd '$repoWsl' && /home/idgal/miniforge3/envs/video_to_world/bin/python -u '$repoWsl/tmp_gradio_server.py'"

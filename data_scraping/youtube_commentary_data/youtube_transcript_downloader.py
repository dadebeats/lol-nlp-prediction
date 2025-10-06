import os
from yt_dlp import YoutubeDL
from youtube_channels import league_to_youtube_channel_mapping

# Target source (e.g. playlist, channel, or video)

output_dir = 'commentary_data'

# Loop over leagues
for league_id, url in league_to_youtube_channel_mapping.items():
    if url:
        # Create a league-specific folder
        league_folder = os.path.join(output_dir, league_id)
        if os.path.exists(league_folder):
            print(f"Skipping league {league_id} (already downloaded)")
            continue
        os.makedirs(league_folder, exist_ok=True)

        # Configure yt-dlp with a league-specific outtmpl
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'writeinfojson': True,
            'outtmpl': os.path.join(league_folder, '%(title)s.%(ext)s'),
            'quiet': False,
            'nocheckcertificate': True,
            'subtitlesformat': 'srt',
            'overwrites': False,
            'ignoreerrors': True,
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])


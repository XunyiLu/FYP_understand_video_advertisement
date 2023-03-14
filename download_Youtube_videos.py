from pytube import YouTube


def download_youtube_videos(csv, target_path):
    # data = ' '
    with open(csv, 'r') as f:
        for line in f.readlines():
            line = line.replace("'", "")
            data = 'https://www.youtube.com/watch?v=' + line
            print(data)
            yt = YouTube(data)

            try:
                # print(yt.)
                yt.streams.get_highest_resolution().download(target_path, filename=line + '.mp4')
                # video.download('/User/lxy/Documents/')
                # print(data)

            except:
                print("An exception occurred")


download_youtube_videos("/Users/bunnylu/Desktop/FYP/final_video_id_list_01.csv",
                        "/Users/bunnylu/Desktop/FYP/database/video/")



# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    print('Hi', name)  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    from pytube import YouTube
    data = ' '
    with open('/Users/lxy/Desktop/FYP/final_video_id_list.csv', 'r') as f:
            for line in f.readlines():
                line = line.replace("'","")
                data = 'https://www.youtube.com/watch?v=' + line
                print(data)
                yt = YouTube(data)

                try:
                    # print(yt.)
                    yt.streams.get_highest_resolution().download('/Users/lxy/Desktop/FYP/videos/new/', filename = "_v_exciting_"+ line + '.mp4')
                    # video.download('/User/lxy/Documents/')
                    # print(data)


                except:
                    print("An exception occurred")


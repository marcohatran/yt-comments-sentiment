from googleapiclient.discovery import build


class Comment:
    """ Description of Service """

    def __init__(self, youtube_api_key: str, video_id: str):
        """ Description of Method """
        self.video_id = video_id
        self.service = build('youtube', 'v3', developerKey=youtube_api_key)

    def build_initial_request(self, video_id):
        request = self.service.commentThreads().list(part="snippet", videoId=video_id)
        return request

    def parse_page(self, result, all_comments):
        for item in result["items"]:
            # print(item['snippet']['topLevelComment']['snippet']['authorDisplayName'])
            # print('---')
            comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            # print(comment_text)
            # print(item['snippet']['topLevelComment']['snippet']['textOriginal'])
            all_comments.append(comment_text)
            # print('====================')

    def parse_result(self, result, all_comments, video_id):
        # 50 comments only to avoid quotaExceeded error
        while result.get("nextPageToken", False) and len(all_comments) <= 50:
            self.parse_page(result, all_comments)
            request = self.service.commentThreads().list(
                part="snippet", videoId=video_id, pageToken=result.get("nextPageToken")
            )
            result = request.execute()

    def get_all_comments(self):
        # Build request and fetch comments
        all_comments = list()
        request = self.build_initial_request(self.video_id)
        result = request.execute()
        self.parse_result(result, all_comments, self.video_id)
        return all_comments

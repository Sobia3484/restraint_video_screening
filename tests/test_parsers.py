import unittest

from vidscreen.extractors.tiktok import parse_aweme
from vidscreen.queries import Query
from vidscreen.transcripts import caption_infos, parse_webvtt, pick_caption
from vidscreen.extractors.youtube import youtube_record

Q = Query("Q002", "Infant CPR", "cpr", "en")

AWEME = {  # shape follows the ScrapeCreators keyword-search sample
    "aweme_id": "7334621391758642478",
    "desc": "How to do infant CPR #infantcpr #FirstAid #infantcpr",
    "create_time": 1707724682,
    "region": "US",
    "desc_language": "en",
    "author": {"unique_id": "nurse_jo", "nickname": "Nurse Jo", "follower_count": 1200},
    "statistics": {"play_count": 31677, "digg_count": 355, "comment_count": 5, "share_count": 22, "collect_count": 30},
    "text_extra": [{"hashtag_name": "infantcpr"}, {"hashtag_name": "FirstAid"}],
    "cha_list": [{"cha_name": "cprtraining"}],
    "video": {
        "duration": 83851,
        "cover": {"url_list": ["https://example.com/c.jpeg"]},
        "cla_info": {"caption_infos": [
            {"language_code": "en", "is_original_caption": True, "is_auto_generated": True, "url": "https://example.com/en.vtt"},
            {"language_code": "ur", "is_original_caption": False, "url": "https://example.com/ur.vtt"},
        ]},
    },
}


class TikTokParseTests(unittest.TestCase):
    def test_parse_fields(self):
        r = parse_aweme(Q, 3, AWEME)
        self.assertEqual(r.platform, "tiktok")
        self.assertEqual(r.video_id, "7334621391758642478")
        self.assertEqual(r.url, "https://www.tiktok.com/@nurse_jo/video/7334621391758642478")
        self.assertEqual(r.duration_seconds, 83.85)
        self.assertEqual((r.views, r.likes, r.comments, r.shares, r.saves), (31677, 355, 5, 22, 30))
        self.assertEqual(r.upload_date, "2024-02-12")
        self.assertEqual(r.hashtags, "infantcpr|firstaid|cprtraining")
        self.assertEqual((r.country_code, r.country_source), ("US", "tiktok_video_region"))
        self.assertEqual(r.query_rank, 3)

    def test_missing_fields_are_blank_not_crash(self):
        r = parse_aweme(Q, 1, {"aweme_id": 99})
        self.assertEqual(r.video_id, "99")
        self.assertIsNone(r.views)
        self.assertEqual(r.country_code, "")


class CaptionTests(unittest.TestCase):
    def test_pick_prefers_original(self):
        chosen = pick_caption(caption_infos(AWEME), ["ur", "en"])
        self.assertEqual(chosen["language_code"], "en")  # original beats preferred-language translation

    def test_webvtt(self):
        vtt = "WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\nCheck <c>responsiveness</c>\n\n2\n00:00:02.000 --> 00:00:04.000\nCheck responsiveness\n\n3\n00:00:04.000 --> 00:00:05.000\nCall 911 &amp; start CPR\n"
        self.assertEqual(parse_webvtt(vtt), "Check responsiveness Call 911 & start CPR")

    def test_webvtt_keeps_spoken_numbers(self):
        vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n30\n"
        self.assertEqual(parse_webvtt(vtt), "30")


class YouTubeRecordTests(unittest.TestCase):
    def test_ok_record(self):
        info = {"_status": "ok", "id": "abc123", "title": "Infant CPR #cpr", "description": "Steps #firstaid",
                "channel": "Med Ed", "channel_id": "UC1", "duration": 325, "view_count": 1000,
                "like_count": 50, "upload_date": "20240315", "tags": ["cpr", "infant"], "categories": ["Education"]}
        r = youtube_record(Q, 1, info)
        self.assertEqual(r.url, "https://www.youtube.com/watch?v=abc123")
        self.assertEqual(r.upload_date, "2024-03-15")
        self.assertEqual(r.hashtags, "cpr|firstaid")
        self.assertEqual(r.tags, "cpr|infant")
        self.assertEqual(r.metadata_status, "ok")

    def test_unavailable_record_keeps_identity(self):
        r = youtube_record(Q, 2, {"id": "zzz", "_status": "unavailable", "_error": "Video unavailable"})
        self.assertEqual(r.video_id, "zzz")
        self.assertTrue(r.metadata_status.startswith("unavailable"))


if __name__ == "__main__":
    unittest.main()

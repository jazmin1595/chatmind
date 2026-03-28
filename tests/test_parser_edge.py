"""Parser edge case tests - malformed data, encoding, special characters.

50 tests covering real-world messy data that parsers must handle.
"""

import os
import json
import csv
import tempfile
from datetime import datetime

import pytest

from chatmind.parsers.discord import parse_discord_json, parse_discord_csv, parse_discord
from chatmind.parsers.kakao import parse_kakao
from chatmind.parsers.auto import auto_parse, _detect_platform


# =========================================================
#  DISCORD JSON EDGE CASES (18)
# =========================================================
class TestDiscordJsonEdge:
    def test_unicode_emoji_in_content(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "Let's gooo! 🎮🔥💯", "author": {"name": "user1"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert len(msgs) == 1
        assert "🎮" in msgs[0].content

    def test_korean_content(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "오늘 치킨 먹자 ㅋㅋㅋ", "author": {"name": "김철수"}},
        ], "channel": {"name": "잡담"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert msgs[0].content == "오늘 치킨 먹자 ㅋㅋㅋ"
        assert msgs[0].sender == "김철수"
        assert msgs[0].room == "잡담"

    def test_very_long_content(self):
        long_text = "A" * 5000
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": long_text, "author": {"name": "user"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert len(msgs[0].content) == 5000

    def test_special_characters_in_name(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "hello", "author": {"name": "user [BOT] #1234"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert "BOT" in msgs[0].sender

    def test_missing_author_field(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "hello", "author": {}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert len(msgs) == 1
        assert msgs[0].sender == "Unknown"

    def test_missing_channel_field(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "hello", "author": {"name": "user"}},
        ]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert msgs[0].room == "unknown"

    def test_whitespace_only_content_skipped(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "   \n\t  ", "author": {"name": "user"}},
            {"id": "2", "timestamp": "2024-01-01T00:01:00+00:00",
             "content": "real message", "author": {"name": "user"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert len(msgs) == 1

    def test_timestamp_with_milliseconds(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-06-15T13:45:30.123456+00:00",
             "content": "msg", "author": {"name": "user"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert msgs[0].timestamp.year == 2024
        assert msgs[0].timestamp.month == 6

    def test_timestamp_with_Z(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00Z",
             "content": "msg", "author": {"name": "user"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert len(msgs) == 1

    def test_timestamp_kst_timezone(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T09:00:00+09:00",
             "content": "msg", "author": {"name": "user"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert len(msgs) == 1

    def test_bad_timestamp_skipped(self):
        data = {"messages": [
            {"id": "1", "timestamp": "not-a-date",
             "content": "msg", "author": {"name": "user"}},
            {"id": "2", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "good", "author": {"name": "user"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert len(msgs) == 1
        assert msgs[0].content == "good"

    def test_empty_timestamp_skipped(self):
        data = {"messages": [
            {"id": "1", "timestamp": "",
             "content": "msg", "author": {"name": "user"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert len(msgs) == 0

    def test_newlines_in_content(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "line1\nline2\nline3", "author": {"name": "user"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert "\n" in msgs[0].content

    def test_url_in_content(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "Check this out https://github.com/repo/project", "author": {"name": "user"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert "https://" in msgs[0].content

    def test_many_messages(self):
        """100 messages in one file."""
        msg_list = []
        for i in range(100):
            msg_list.append({
                "id": str(i), "timestamp": f"2024-01-01T{i//60:02d}:{i%60:02d}:00+00:00",
                "content": f"Message number {i}", "author": {"name": f"user{i%5}"}
            })
        data = {"messages": msg_list, "channel": {"name": "bulk"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert len(msgs) == 100

    def test_nickname_fallback(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "hi", "author": {"name": "", "nickname": "NickName123"}},
        ], "channel": {"name": "test"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert msgs[0].sender == "NickName123"

    def test_data_package_string_author(self):
        """Discord data package where Author is a string, not dict."""
        data = [
            {"ID": "1", "Timestamp": "2024-01-01T00:00:00+00:00",
             "Contents": "hello", "Author": "simple_username"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = parse_discord_json(f.name)
        os.unlink(f.name)
        assert msgs[0].sender == "simple_username"

    def test_multiple_json_in_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, name in enumerate(["ch1.json", "ch2.json"]):
                data = {"messages": [
                    {"id": str(i), "timestamp": "2024-01-01T00:00:00+00:00",
                     "content": f"msg from {name}", "author": {"name": "user"}},
                ], "channel": {"name": f"channel{i}"}}
                with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as f:
                    json.dump(data, f)
            msgs = parse_discord(tmpdir)
            assert len(msgs) == 2


# =========================================================
#  DISCORD CSV EDGE CASES (6)
# =========================================================
class TestDiscordCsvEdge:
    def test_basic_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                         encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["AuthorID", "Author", "Date", "Content"])
            writer.writeheader()
            writer.writerow({"AuthorID": "1", "Author": "user1",
                             "Date": "2024-01-01T10:00:00+00:00", "Content": "hello"})
            writer.writerow({"AuthorID": "2", "Author": "user2",
                             "Date": "2024-01-01T10:01:00+00:00", "Content": "world"})
            f.flush()
            msgs = parse_discord_csv(f.name)
        os.unlink(f.name)
        assert len(msgs) == 2

    def test_csv_empty_content_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                         encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["AuthorID", "Author", "Date", "Content"])
            writer.writeheader()
            writer.writerow({"AuthorID": "1", "Author": "user",
                             "Date": "2024-01-01T10:00:00+00:00", "Content": ""})
            f.flush()
            msgs = parse_discord_csv(f.name)
        os.unlink(f.name)
        assert len(msgs) == 0

    def test_csv_with_commas_in_content(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                         encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["AuthorID", "Author", "Date", "Content"])
            writer.writeheader()
            writer.writerow({"AuthorID": "1", "Author": "user",
                             "Date": "2024-01-01T10:00:00+00:00",
                             "Content": "hello, world, how are you?"})
            f.flush()
            msgs = parse_discord_csv(f.name)
        os.unlink(f.name)
        assert len(msgs) == 1
        assert "hello, world" in msgs[0].content

    def test_csv_bad_timestamp_skipped(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                         encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["AuthorID", "Author", "Date", "Content"])
            writer.writeheader()
            writer.writerow({"AuthorID": "1", "Author": "user",
                             "Date": "INVALID", "Content": "msg"})
            f.flush()
            msgs = parse_discord_csv(f.name)
        os.unlink(f.name)
        assert len(msgs) == 0

    def test_csv_auto_detected_as_discord(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                         encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["AuthorID", "Author", "Date", "Content"])
            writer.writeheader()
            writer.writerow({"AuthorID": "1", "Author": "user",
                             "Date": "2024-01-01T10:00:00+00:00", "Content": "test"})
            f.flush()
            platform = _detect_platform(f.name)
        os.unlink(f.name)
        assert platform == "discord"

    def test_csv_room_override(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False,
                                         encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["AuthorID", "Author", "Date", "Content"])
            writer.writeheader()
            writer.writerow({"AuthorID": "1", "Author": "user",
                             "Date": "2024-01-01T10:00:00+00:00", "Content": "test"})
            f.flush()
            msgs = parse_discord_csv(f.name, room="my-channel")
        os.unlink(f.name)
        assert msgs[0].room == "my-channel"


# =========================================================
#  KAKAOTALK EDGE CASES (16)
# =========================================================
class TestKakaoEdge:
    def _write_and_parse(self, content):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            msgs = parse_kakao(f.name)
        os.unlink(f.name)
        return msgs

    def test_midnight_am(self):
        msgs = self._write_and_parse("""test
--------------- 2024년 1월 1일 월요일 ---------------
[user] 오전 12:00 자정 메시지
""")
        assert len(msgs) == 1
        assert msgs[0].timestamp.hour == 0

    def test_noon_pm(self):
        msgs = self._write_and_parse("""test
--------------- 2024년 1월 1일 월요일 ---------------
[user] 오후 12:00 정오 메시지
""")
        assert len(msgs) == 1
        assert msgs[0].timestamp.hour == 12

    def test_pm_11(self):
        msgs = self._write_and_parse("""test
--------------- 2024년 1월 1일 월요일 ---------------
[user] 오후 11:59 밤 메시지
""")
        assert len(msgs) == 1
        assert msgs[0].timestamp.hour == 23
        assert msgs[0].timestamp.minute == 59

    def test_multiple_dates(self):
        msgs = self._write_and_parse("""test
--------------- 2024년 1월 1일 월요일 ---------------
[A] 오전 9:00 첫날 메시지
--------------- 2024년 1월 2일 화요일 ---------------
[B] 오후 3:00 둘째날 메시지
""")
        assert len(msgs) == 2
        assert msgs[0].timestamp.day == 1
        assert msgs[1].timestamp.day == 2

    def test_multiple_messages_same_user(self):
        msgs = self._write_and_parse("""test
--------------- 2024년 1월 1일 월요일 ---------------
[김철수] 오전 10:00 첫번째
[김철수] 오전 10:01 두번째
[김철수] 오전 10:02 세번째
""")
        assert len(msgs) == 3
        assert all(m.sender == "김철수" for m in msgs)

    def test_emoji_in_content(self):
        msgs = self._write_and_parse("""test
--------------- 2024년 1월 1일 월요일 ---------------
[user] 오전 10:00 ㅋㅋㅋ 😂🔥
""")
        assert len(msgs) == 1
        assert "😂" in msgs[0].content

    def test_url_in_content(self):
        msgs = self._write_and_parse("""test
--------------- 2024년 1월 1일 월요일 ---------------
[user] 오전 10:00 https://www.youtube.com/watch?v=abc123 이거 봐
""")
        assert len(msgs) == 1
        assert "youtube.com" in msgs[0].content

    def test_long_sender_name(self):
        msgs = self._write_and_parse("""test
--------------- 2024년 1월 1일 월요일 ---------------
[아주 긴 닉네임을 가진 사용자 이름] 오전 10:00 안녕하세요
""")
        assert len(msgs) == 1
        assert msgs[0].sender == "아주 긴 닉네임을 가진 사용자 이름"

    def test_system_messages_ignored(self):
        """Non-message lines (system alerts) should be ignored."""
        msgs = self._write_and_parse("""채팅방
--------------- 2024년 1월 1일 월요일 ---------------
김철수님이 들어왔습니다.
[김철수] 오전 10:00 안녕
이영희님이 들어왔습니다.
[이영희] 오전 10:01 반가워
""")
        assert len(msgs) == 2

    def test_empty_content_skipped(self):
        """Messages with only whitespace after sender info."""
        msgs = self._write_and_parse("""test
--------------- 2024년 1월 1일 월요일 ---------------
[user] 오전 10:00 실제 메시지
""")
        assert len(msgs) == 1

    def test_no_date_header_no_crash(self):
        """Messages before any date header should be ignored."""
        msgs = self._write_and_parse("""채팅방 이름
[user] 오전 10:00 날짜 없는 메시지
""")
        assert len(msgs) == 0

    def test_room_from_first_line(self):
        msgs = self._write_and_parse("""우리친구들 단톡방
--------------- 2024년 1월 1일 월요일 ---------------
[김철수] 오전 10:00 안녕
""")
        # Room auto-detected from first line
        assert len(msgs) == 1

    def test_many_kakao_messages(self):
        """50 messages."""
        lines = ["test\n"]
        lines.append("--------------- 2024년 1월 1일 월요일 ---------------\n")
        for i in range(50):
            h = 9 + i // 60
            m = i % 60
            if h < 12:
                lines.append(f"[user{i%5}] 오전 {h}:{m:02d} 메시지 {i}\n")
            else:
                hh = h - 12 if h > 12 else 12
                lines.append(f"[user{i%5}] 오후 {hh}:{m:02d} 메시지 {i}\n")
        msgs = self._write_and_parse("".join(lines))
        assert len(msgs) == 50

    def test_english_format_kakao(self):
        """KakaoTalk English format."""
        msgs = self._write_and_parse("""Chat Room
--------------- Friday, March 15, 2024 ---------------
[John] 2:30 PM Hello everyone!
[Jane] 2:31 PM Hi there
""")
        assert len(msgs) == 2
        assert msgs[0].sender == "John"
        assert msgs[0].timestamp.hour == 14

    def test_english_am_format(self):
        msgs = self._write_and_parse("""Chat
--------------- Monday, January 1, 2024 ---------------
[user] 9:00 AM Good morning
""")
        assert len(msgs) == 1
        assert msgs[0].timestamp.hour == 9

    def test_english_midnight(self):
        msgs = self._write_and_parse("""Chat
--------------- Monday, January 1, 2024 ---------------
[user] 12:00 AM Midnight message
""")
        assert len(msgs) == 1
        assert msgs[0].timestamp.hour == 0


# =========================================================
#  AUTO-DETECT EDGE CASES (10)
# =========================================================
class TestAutoDetectEdge:
    def test_detect_directory_with_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test.json")
            with open(fpath, "w") as f:
                json.dump({"messages": []}, f)
            assert _detect_platform(tmpdir) == "discord"

    def test_detect_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _detect_platform(tmpdir) == "unknown"

    def test_auto_parse_raises_on_unknown(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"random content")
            f.flush()
        with pytest.raises(ValueError, match="Cannot detect"):
            auto_parse(f.name)
        os.unlink(f.name)

    def test_auto_parse_explicit_discord(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "test", "author": {"name": "user"}},
        ], "channel": {"name": "ch"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = auto_parse(f.name, platform="discord")
        os.unlink(f.name)
        assert len(msgs) == 1

    def test_auto_parse_explicit_kakao(self):
        content = """test
--------------- 2024년 1월 1일 월요일 ---------------
[user] 오전 10:00 안녕
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            msgs = auto_parse(f.name, platform="kakao")
        os.unlink(f.name)
        assert len(msgs) == 1

    def test_auto_parse_room_override(self):
        data = {"messages": [
            {"id": "1", "timestamp": "2024-01-01T00:00:00+00:00",
             "content": "test", "author": {"name": "user"}},
        ], "channel": {"name": "original"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            msgs = auto_parse(f.name, room="override-room")
        os.unlink(f.name)
        assert msgs[0].room == "override-room"

    def test_detect_discord_list_format(self):
        data = [{"ID": "1", "Contents": "hi", "Timestamp": "2024-01-01T00:00:00",
                 "Author": {"username": "u"}}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            platform = _detect_platform(f.name)
        os.unlink(f.name)
        assert platform == "discord"

    def test_detect_txt_with_korean_markers(self):
        content = """chat
--------------- 2024년 3월 1일 금요일 ---------------
[user] 오후 1:00 test
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            platform = _detect_platform(f.name)
        os.unlink(f.name)
        assert platform == "kakao"

    def test_detect_plain_txt_defaults_kakao(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("just some text without markers\n" * 25)
            f.flush()
            platform = _detect_platform(f.name)
        os.unlink(f.name)
        assert platform == "kakao"

    def test_detect_csv_as_discord(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("Author,Content\nuser,hello\n")
            f.flush()
            platform = _detect_platform(f.name)
        os.unlink(f.name)
        assert platform == "discord"

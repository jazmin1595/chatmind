"""Integration tests using REAL sentence-transformers model.

No mocks - actual embeddings from paraphrase-multilingual-MiniLM-L12-v2.
Tests realistic Discord messages including Korean, slang, emoji, mixed language.

These tests are slower (~30s) but validate real-world accuracy.
"""

import pytest
import numpy as np
from datetime import datetime

from chatmind.models import ChatMessage, ChatIndex
from chatmind.searcher import search
from chatmind.indexer import build_index


# ---- Realistic Discord messages (Korean + English + slang + emoji) ----
MESSAGES = [
    # Food (0-7)
    ChatMessage(datetime(2024, 1, 10, 12, 0), "민수", "강남역 스시오마카세 진짜 미쳤음 코스 5만원인데 퀄리티 개좋아", "맛집추천", "discord"),
    ChatMessage(datetime(2024, 1, 10, 12, 1), "영희", "홍대 이자카야 새로 생긴데 가봤는데 사시미가 신선함", "맛집추천", "discord"),
    ChatMessage(datetime(2024, 1, 10, 12, 2), "철수", "어제 을지로 노포 갈비집 웨이팅 1시간 했는데 worth it", "맛집추천", "discord"),
    ChatMessage(datetime(2024, 1, 10, 12, 3), "지은", "파스타 맛집 찾는데 이태원쪽 괜찮은데 있어?", "맛집추천", "discord"),
    ChatMessage(datetime(2024, 1, 10, 12, 4), "민수", "떡볶이는 역시 신당동 즉떡이 원탑임 ㅋㅋ", "맛집추천", "discord"),
    ChatMessage(datetime(2024, 1, 10, 12, 5), "영희", "카페 디저트 추천좀 티라미수 맛있는데", "맛집추천", "discord"),
    ChatMessage(datetime(2024, 1, 10, 12, 6), "Alex", "That ramen place near the station is absolutely fire ngl", "맛집추천", "discord"),
    ChatMessage(datetime(2024, 1, 10, 12, 7), "철수", "치킨은 교촌 vs bhc 뭐가 더 맛있음?", "맛집추천", "discord"),

    # Gaming (8-15)
    ChatMessage(datetime(2024, 1, 11, 20, 0), "민수", "오늘 밤 롤 한판 ㄱ? 골드 승급전임", "게임", "discord"),
    ChatMessage(datetime(2024, 1, 11, 20, 1), "철수", "발로란트 새 시즌 패치 나왔는데 제트 너프 ㅠㅠ", "게임", "discord"),
    ChatMessage(datetime(2024, 1, 11, 20, 2), "지은", "스팀 여름 세일에 게임 10개 샀음 ㅋㅋㅋ", "게임", "discord"),
    ChatMessage(datetime(2024, 1, 11, 20, 3), "Alex", "anyone down for minecraft? just built a sick base", "게임", "discord"),
    ChatMessage(datetime(2024, 1, 11, 20, 4), "민수", "RTX 4070 질렀다 배그 울트라 144프레임 나옴", "게임", "discord"),
    ChatMessage(datetime(2024, 1, 11, 20, 5), "영희", "젤다 티어킨 엔딩 봤는데 개감동이야 ㅠㅠ", "게임", "discord"),
    ChatMessage(datetime(2024, 1, 11, 20, 6), "철수", "서버 렉 장난 아닌데 램 업그레이드 해야할듯", "게임", "discord"),
    ChatMessage(datetime(2024, 1, 11, 20, 7), "지은", "PS5 독점작 갓오워 라그나로크 시작했음", "게임", "discord"),

    # Study (16-23)
    ChatMessage(datetime(2024, 1, 12, 14, 0), "영희", "미적분 과제 3번 어떻게 풀어? 막혀서 죽겠음", "공부", "discord"),
    ChatMessage(datetime(2024, 1, 12, 14, 1), "철수", "파이썬 알고리즘 시험 내일인데 정렬 아직도 모르겠어", "공부", "discord"),
    ChatMessage(datetime(2024, 1, 12, 14, 2), "민수", "도서관 3층에 자리 있어 같이 공부하자", "공부", "discord"),
    ChatMessage(datetime(2024, 1, 12, 14, 3), "지은", "물리 실험 보고서 20장이래 ㅋㅋ 미친건가", "공부", "discord"),
    ChatMessage(datetime(2024, 1, 12, 14, 4), "Alex", "organic chem midterm was brutal, think I failed lol", "공부", "discord"),
    ChatMessage(datetime(2024, 1, 12, 14, 5), "영희", "장학금 신청 마감 금요일까지인데 추천서 받았어?", "공부", "discord"),
    ChatMessage(datetime(2024, 1, 12, 14, 6), "철수", "머신러닝 기말 프로젝트 주제 뭘로 할지 고민중", "공부", "discord"),
    ChatMessage(datetime(2024, 1, 12, 14, 7), "민수", "영어 토익 900 넘기고 싶은데 팁 있어?", "공부", "discord"),

    # Travel (24-31)
    ChatMessage(datetime(2024, 1, 13, 10, 0), "지은", "봄에 제주도 가자! 3박4일 렌트카 빌리면 될듯", "여행", "discord"),
    ChatMessage(datetime(2024, 1, 13, 10, 1), "민수", "에어비앤비 한라산 근처로 예약했어 경치 대박", "여행", "discord"),
    ChatMessage(datetime(2024, 1, 13, 10, 2), "영희", "오사카 항공권 왕복 20만원 떴다!! 예약 ㄱㄱ", "여행", "discord"),
    ChatMessage(datetime(2024, 1, 13, 10, 3), "Alex", "backpacking through Europe this summer, any tips?", "여행", "discord"),
    ChatMessage(datetime(2024, 1, 13, 10, 4), "철수", "방콕 풀빌라 가격 생각보다 싸더라 1박 10만원", "여행", "discord"),
    ChatMessage(datetime(2024, 1, 13, 10, 5), "지은", "파리 에펠탑 야경 사진 올릴게 너무 예뻤어", "여행", "discord"),
    ChatMessage(datetime(2024, 1, 13, 10, 6), "민수", "여행 짐 싸는중인데 캐리어 사이즈 뭐가 좋아?", "여행", "discord"),
    ChatMessage(datetime(2024, 1, 13, 10, 7), "영희", "해외여행 보험 들어야 하나? 추천 있으면 알려줘", "여행", "discord"),

    # Daily/Random (32-39)
    ChatMessage(datetime(2024, 1, 14, 9, 0), "철수", "오늘 날씨 개춥다 -15도래 ㄷㄷ", "잡담", "discord"),
    ChatMessage(datetime(2024, 1, 14, 9, 1), "지은", "새로 나온 아이폰 16 프로 사야하나 말아야하나", "잡담", "discord"),
    ChatMessage(datetime(2024, 1, 14, 9, 2), "Alex", "my cat knocked over my coffee this morning lmao", "잡담", "discord"),
    ChatMessage(datetime(2024, 1, 14, 9, 3), "민수", "BTS 콘서트 티켓팅 실패했어 ㅠㅠ 2초만에 매진", "잡담", "discord"),
    ChatMessage(datetime(2024, 1, 14, 9, 4), "영희", "넷플릭스 뭐 볼만한거 있어? 추천좀", "잡담", "discord"),
    ChatMessage(datetime(2024, 1, 14, 9, 5), "철수", "헬스 3개월째인데 벤치프레스 80키로 됐다!", "잡담", "discord"),
    ChatMessage(datetime(2024, 1, 14, 9, 6), "지은", "알바 시급 만이천원인데 너무 적은거 아니야?", "잡담", "discord"),
    ChatMessage(datetime(2024, 1, 14, 9, 7), "민수", "주말에 한강 치맥 ㄱ? 날씨 좋으면", "잡담", "discord"),
]

FOOD_INDICES = list(range(0, 8))
GAMING_INDICES = list(range(8, 16))
STUDY_INDICES = list(range(16, 24))
TRAVEL_INDICES = list(range(24, 32))
DAILY_INDICES = list(range(32, 40))


@pytest.fixture(scope="module")
def real_index():
    """Build index with REAL sentence-transformers model (runs once)."""
    index = build_index(
        MESSAGES,
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        bits=3,
        batch_size=64,
    )
    return index


@pytest.fixture(scope="module")
def real_model():
    """Load real model once for all tests."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# =========================================================
#  FOOD SEARCH - Korean + English food queries
# =========================================================
class TestRealFood:
    def test_korean_food_query(self, real_index, real_model):
        """한국어 맛집 검색"""
        r = search("맛집 추천해줘", real_index, k=5, model=real_model)
        top3_idx = [MESSAGES.index(x.message) for x in r[:3]]
        assert any(i in FOOD_INDICES for i in top3_idx)

    def test_english_food_query(self, real_index, real_model):
        """영어로 음식 검색"""
        r = search("restaurant recommendation", real_index, k=5, model=real_model)
        top3_idx = [MESSAGES.index(x.message) for x in r[:3]]
        assert any(i in FOOD_INDICES for i in top3_idx)

    def test_sushi_search(self, real_index, real_model):
        """스시/초밥 검색"""
        r = search("스시 초밥 맛있는 곳", real_index, k=3, model=real_model)
        top3_idx = [MESSAGES.index(x.message) for x in r[:3]]
        assert any(i in FOOD_INDICES for i in top3_idx)

    def test_chicken_search(self, real_index, real_model):
        """치킨 관련 검색"""
        r = search("치킨 추천", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in FOOD_INDICES for i in top5_idx)

    def test_food_top5_has_food(self, real_index, real_model):
        """맛집 검색 시 top5에 음식 메시지 2개 이상"""
        r = search("저녁 뭐 먹지 맛있는거", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        food_count = sum(1 for i in top5_idx if i in FOOD_INDICES)
        assert food_count >= 2

    def test_cross_language_food(self, real_index, real_model):
        """영어 쿼리로 한국어 맛집 메시지 찾기"""
        r = search("best Korean BBQ restaurant", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in FOOD_INDICES for i in top5_idx)


# =========================================================
#  GAMING SEARCH
# =========================================================
class TestRealGaming:
    def test_korean_gaming_query(self, real_index, real_model):
        """한국어 게임 검색"""
        r = search("게임 같이 하자", real_index, k=5, model=real_model)
        top3_idx = [MESSAGES.index(x.message) for x in r[:3]]
        assert any(i in GAMING_INDICES for i in top3_idx)

    def test_lol_search(self, real_index, real_model):
        """롤/리그오브레전드 검색"""
        r = search("롤 랭크 게임", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in GAMING_INDICES for i in top5_idx)

    def test_gpu_search(self, real_index, real_model):
        """그래픽카드 검색"""
        r = search("그래픽카드 GPU 업그레이드", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in GAMING_INDICES for i in top5_idx)

    def test_english_gaming(self, real_index, real_model):
        """영어 게임 검색"""
        r = search("let's play games tonight", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in GAMING_INDICES for i in top5_idx)

    def test_server_lag(self, real_index, real_model):
        """서버 렉 관련 검색"""
        r = search("서버 렉 걸려서 느려", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in GAMING_INDICES for i in top5_idx)

    def test_gaming_not_study(self, real_index, real_model):
        """게임 검색 시 공부 메시지가 1위가 아닌지"""
        r = search("게임 플레이 새 시즌", real_index, k=1, model=real_model)
        idx = MESSAGES.index(r[0].message)
        assert idx not in STUDY_INDICES


# =========================================================
#  STUDY SEARCH
# =========================================================
class TestRealStudy:
    def test_homework_search(self, real_index, real_model):
        """과제 검색"""
        r = search("과제 도와줘 숙제", real_index, k=5, model=real_model)
        top3_idx = [MESSAGES.index(x.message) for x in r[:3]]
        assert any(i in STUDY_INDICES for i in top3_idx)

    def test_exam_search(self, real_index, real_model):
        """시험 관련 검색"""
        r = search("시험 공부 준비", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in STUDY_INDICES for i in top5_idx)

    def test_library_search(self, real_index, real_model):
        """도서관 검색"""
        r = search("도서관에서 같이 공부", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in STUDY_INDICES for i in top5_idx)

    def test_programming_search(self, real_index, real_model):
        """프로그래밍 검색"""
        r = search("코딩 프로그래밍 알고리즘", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in STUDY_INDICES for i in top5_idx)

    def test_english_study(self, real_index, real_model):
        """영어 시험 검색"""
        r = search("midterm exam failed", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in STUDY_INDICES for i in top5_idx)

    def test_study_not_travel(self, real_index, real_model):
        """공부 검색 시 여행이 1위가 아닌지"""
        r = search("과제 제출 기한 마감", real_index, k=1, model=real_model)
        idx = MESSAGES.index(r[0].message)
        assert idx not in TRAVEL_INDICES


# =========================================================
#  TRAVEL SEARCH
# =========================================================
class TestRealTravel:
    def test_jeju_search(self, real_index, real_model):
        """제주도 여행 검색"""
        r = search("제주도 여행 계획", real_index, k=5, model=real_model)
        top3_idx = [MESSAGES.index(x.message) for x in r[:3]]
        assert any(i in TRAVEL_INDICES for i in top3_idx)

    def test_flight_search(self, real_index, real_model):
        """항공권 검색"""
        r = search("비행기 항공권 싸게", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in TRAVEL_INDICES for i in top5_idx)

    def test_accommodation_search(self, real_index, real_model):
        """숙소 검색"""
        r = search("숙소 에어비앤비 호텔 예약", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in TRAVEL_INDICES for i in top5_idx)

    def test_backpacking_search(self, real_index, real_model):
        """배낭여행 검색"""
        r = search("backpacking Europe budget travel", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in TRAVEL_INDICES for i in top5_idx)

    def test_packing_search(self, real_index, real_model):
        """여행 짐싸기 검색"""
        r = search("여행 짐 캐리어 준비", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in TRAVEL_INDICES for i in top5_idx)

    def test_travel_not_food(self, real_index, real_model):
        """여행 검색 시 맛집이 1위가 아닌지"""
        r = search("해외여행 비행기 호텔", real_index, k=1, model=real_model)
        idx = MESSAGES.index(r[0].message)
        assert idx not in FOOD_INDICES


# =========================================================
#  CROSS-TOPIC / HARD CASES
# =========================================================
class TestRealHardCases:
    def test_slang_korean(self, real_index, real_model):
        """한국어 줄임말/슬랭 이해"""
        r = search("ㄱㄱ 같이 하자", real_index, k=5, model=real_model)
        assert len(r) > 0  # at least returns something

    def test_mixed_language(self, real_index, real_model):
        """한영 혼용 검색"""
        r = search("game 할 사람?", real_index, k=5, model=real_model)
        assert len(r) > 0

    def test_vague_memory(self, real_index, real_model):
        """모호한 기억으로 검색 - 맛집"""
        r = search("저번에 누가 추천해준 일본 음식점", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in FOOD_INDICES for i in top5_idx)

    def test_vague_memory_travel(self, real_index, real_model):
        """모호한 기억으로 검색 - 여행"""
        r = search("누가 비행기 싸다고 했던거", real_index, k=5, model=real_model)
        top5_idx = [MESSAGES.index(x.message) for x in r[:5]]
        assert any(i in TRAVEL_INDICES for i in top5_idx)

    def test_weather_query(self, real_index, real_model):
        """날씨 검색"""
        r = search("오늘 날씨 어때 추워?", real_index, k=3, model=real_model)
        assert any("날씨" in x.message.content or "춥" in x.message.content for x in r[:3])

    def test_concert_query(self, real_index, real_model):
        """콘서트 검색"""
        r = search("콘서트 티켓", real_index, k=3, model=real_model)
        assert any("콘서트" in x.message.content or "티켓" in x.message.content for x in r[:3])

    def test_netflix_query(self, real_index, real_model):
        """넷플릭스/영화 검색 - top5에서 찾기"""
        r = search("넷플릭스 영화 드라마 볼만한거 추천", real_index, k=5, model=real_model)
        assert any("넷플릭스" in x.message.content or "볼만한" in x.message.content for x in r[:5])

    def test_exercise_query(self, real_index, real_model):
        """운동 검색 - 벤치프레스/헬스 메시지를 top10에서 찾기"""
        r = search("헬스장 벤치프레스 웨이트 트레이닝", real_index, k=10, model=real_model)
        assert any("헬스" in x.message.content or "벤치" in x.message.content for x in r[:10])


# =========================================================
#  FILTER TESTS WITH REAL MODEL
# =========================================================
class TestRealFilters:
    def test_filter_sender_korean(self, real_index, real_model):
        """한국어 이름 발신자 필터"""
        r = search("맛집", real_index, k=10, model=real_model, sender="민수")
        for x in r:
            assert x.message.sender == "민수"

    def test_filter_sender_english(self, real_index, real_model):
        """영어 이름 발신자 필터"""
        r = search("game", real_index, k=10, model=real_model, sender="Alex")
        for x in r:
            assert x.message.sender == "Alex"

    def test_filter_room(self, real_index, real_model):
        """채널 필터"""
        r = search("추천", real_index, k=10, model=real_model, room="맛집")
        for x in r:
            assert "맛집" in x.message.room

    def test_filter_date_range(self, real_index, real_model):
        """날짜 필터"""
        r = search("검색", real_index, k=10, model=real_model,
                    after=datetime(2024, 1, 12), before=datetime(2024, 1, 14))
        for x in r:
            assert x.message.timestamp >= datetime(2024, 1, 12)
            assert x.message.timestamp < datetime(2024, 1, 14)


# =========================================================
#  SCORE QUALITY TESTS
# =========================================================
class TestRealScoreQuality:
    def test_relevant_score_higher(self, real_index, real_model):
        """관련 있는 결과가 관련 없는 결과보다 점수가 높은지"""
        r = search("맛집 음식 추천", real_index, k=40, model=real_model)
        food_scores = [x.score for x in r if MESSAGES.index(x.message) in FOOD_INDICES]
        gaming_scores = [x.score for x in r if MESSAGES.index(x.message) in GAMING_INDICES]
        if food_scores and gaming_scores:
            assert max(food_scores) > max(gaming_scores)

    def test_scores_sorted(self, real_index, real_model):
        """결과가 점수 내림차순인지"""
        r = search("여행 계획", real_index, k=10, model=real_model)
        scores = [x.score for x in r]
        assert scores == sorted(scores, reverse=True)

    def test_top1_score_reasonable(self, real_index, real_model):
        """1위 점수가 합리적인 범위인지"""
        r = search("제주도 여행 가고 싶어", real_index, k=1, model=real_model)
        assert 0.2 < r[0].score < 1.0

    def test_all_scores_in_range(self, real_index, real_model):
        """모든 점수가 -1 ~ 1 범위인지"""
        r = search("아무거나 검색", real_index, k=10, model=real_model)
        for x in r:
            assert -1.0 <= x.score <= 1.0

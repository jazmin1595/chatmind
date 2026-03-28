"""Extended real model tests - more topics, Korean slang, cross-language.

40 additional tests using the actual sentence-transformers model.
"""

import pytest
import numpy as np
from datetime import datetime

from chatmind.models import ChatMessage, ChatIndex
from chatmind.searcher import search
from chatmind.indexer import build_index


MESSAGES = [
    # Work/Job (0-5)
    ChatMessage(datetime(2024, 2, 1, 9, 0), "민수", "면접 준비중인데 자소서 첨삭 좀 해줄 수 있어?", "취업", "discord"),
    ChatMessage(datetime(2024, 2, 1, 9, 1), "영희", "연봉 협상 잘 됐어 20% 인상 ㅋㅋ 개이득", "취업", "discord"),
    ChatMessage(datetime(2024, 2, 1, 9, 2), "철수", "야근 또 해야함 ㅠㅠ 프로젝트 마감이 내일이야", "취업", "discord"),
    ChatMessage(datetime(2024, 2, 1, 9, 3), "Alex", "Got an offer from Google, 150k base salary!", "취업", "discord"),
    ChatMessage(datetime(2024, 2, 1, 9, 4), "지은", "재택근무 최고 출퇴근 안 해도 돼서 삶의 질 올라감", "취업", "discord"),
    ChatMessage(datetime(2024, 2, 1, 9, 5), "민수", "이직 고민중인데 스타트업 vs 대기업 어디가 나을까", "취업", "discord"),

    # Health/Fitness (6-11)
    ChatMessage(datetime(2024, 2, 2, 7, 0), "철수", "오늘 헬스장 가서 데드리프트 120kg 성공했다!", "건강", "discord"),
    ChatMessage(datetime(2024, 2, 2, 7, 1), "지은", "다이어트 시작 3주째 벌써 5키로 빠졌어", "건강", "discord"),
    ChatMessage(datetime(2024, 2, 2, 7, 2), "영희", "병원 갔더니 비타민D 부족이래 영양제 먹어야함", "건강", "discord"),
    ChatMessage(datetime(2024, 2, 2, 7, 3), "민수", "러닝 10km 완주했다 한강 코스 추천함", "건강", "discord"),
    ChatMessage(datetime(2024, 2, 2, 7, 4), "Alex", "Been doing yoga every morning, flexibility improved a lot", "건강", "discord"),
    ChatMessage(datetime(2024, 2, 2, 7, 5), "철수", "독감 걸려서 3일째 누워있음 열이 안 내려가", "건강", "discord"),

    # Entertainment (12-17)
    ChatMessage(datetime(2024, 2, 3, 20, 0), "영희", "넷플릭스 더 글로리 시즌2 정주행 중 개꿀잼", "엔터", "discord"),
    ChatMessage(datetime(2024, 2, 3, 20, 1), "민수", "웹툰 추천좀 요즘 볼만한거 없어?", "엔터", "discord"),
    ChatMessage(datetime(2024, 2, 3, 20, 2), "지은", "에스파 새 앨범 들었는데 타이틀곡 중독성 미쳤음", "엔터", "discord"),
    ChatMessage(datetime(2024, 2, 3, 20, 3), "Alex", "Just finished watching Oppenheimer, incredible cinematography", "엔터", "discord"),
    ChatMessage(datetime(2024, 2, 3, 20, 4), "철수", "유튜브 알고리즘이 자꾸 고양이 영상만 보여줘 ㅋㅋ", "엔터", "discord"),
    ChatMessage(datetime(2024, 2, 3, 20, 5), "영희", "이번 주말 방탈출 카페 가자 새로 오픈한 곳 있대", "엔터", "discord"),

    # Shopping/Money (18-23)
    ChatMessage(datetime(2024, 2, 4, 11, 0), "지은", "쿠팡 로켓배송 새벽에 시켰는데 벌써 왔어 대박", "쇼핑", "discord"),
    ChatMessage(datetime(2024, 2, 4, 11, 1), "민수", "에어팟 맥스 질렀다 가격은 좀 비싸지만 음질 개좋음", "쇼핑", "discord"),
    ChatMessage(datetime(2024, 2, 4, 11, 2), "철수", "무신사 할인 50%라서 옷 5벌 샀음 ㅋㅋ", "쇼핑", "discord"),
    ChatMessage(datetime(2024, 2, 4, 11, 3), "Alex", "Is the new Galaxy S24 worth the upgrade from S23?", "쇼핑", "discord"),
    ChatMessage(datetime(2024, 2, 4, 11, 4), "영희", "이번 달 카드값 150만원 나왔다 ㅋㅋ 거지됐음", "쇼핑", "discord"),
    ChatMessage(datetime(2024, 2, 4, 11, 5), "지은", "당근마켓에서 자전거 3만원에 득템!", "쇼핑", "discord"),

    # Relationship/Social (24-29)
    ChatMessage(datetime(2024, 2, 5, 22, 0), "철수", "소개팅 했는데 분위기 어색해서 죽는줄 알았어", "연애", "discord"),
    ChatMessage(datetime(2024, 2, 5, 22, 1), "영희", "남친이랑 100일 기념으로 편지 썼어 감동받더라", "연애", "discord"),
    ChatMessage(datetime(2024, 2, 5, 22, 2), "지은", "친구들이랑 한강 피크닉 너무 좋았어 날씨도 완벽", "연애", "discord"),
    ChatMessage(datetime(2024, 2, 5, 22, 3), "Alex", "Long distance is tough but we video call every night", "연애", "discord"),
    ChatMessage(datetime(2024, 2, 5, 22, 4), "민수", "동창회 갔다왔는데 10년만에 만나니까 신기해", "연애", "discord"),
    ChatMessage(datetime(2024, 2, 5, 22, 5), "철수", "헤어졌다 ㅠㅠ 3년 사귄 건데 힘들다", "연애", "discord"),
]

WORK_IDX = list(range(0, 6))
HEALTH_IDX = list(range(6, 12))
ENTERTAIN_IDX = list(range(12, 18))
SHOPPING_IDX = list(range(18, 24))
SOCIAL_IDX = list(range(24, 30))


@pytest.fixture(scope="module")
def idx():
    return build_index(
        MESSAGES,
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        bits=3,
    )


@pytest.fixture(scope="module")
def model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# ===== WORK (6) =====
class TestRealWork:
    def test_interview(self, idx, model):
        r = search("면접 자소서 취업 준비", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in WORK_IDX for x in r[:3])

    def test_salary(self, idx, model):
        r = search("연봉 월급 인상", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in WORK_IDX for x in r[:3])

    def test_overtime(self, idx, model):
        r = search("야근 프로젝트 마감", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in WORK_IDX for x in r[:3])

    def test_job_offer_english(self, idx, model):
        r = search("job offer tech company", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in WORK_IDX for x in r[:5])

    def test_remote_work(self, idx, model):
        r = search("재택근무 워라밸", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in WORK_IDX for x in r[:3])

    def test_career_change(self, idx, model):
        r = search("이직 고민 회사", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in WORK_IDX for x in r[:3])


# ===== HEALTH (6) =====
class TestRealHealth:
    def test_gym(self, idx, model):
        r = search("헬스 운동 웨이트", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in HEALTH_IDX for x in r[:3])

    def test_diet(self, idx, model):
        r = search("다이어트 살빼기 체중", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in HEALTH_IDX for x in r[:3])

    def test_hospital(self, idx, model):
        r = search("병원 진료 건강검진", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in HEALTH_IDX for x in r[:5])

    def test_running(self, idx, model):
        r = search("달리기 마라톤 러닝", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in HEALTH_IDX for x in r[:3])

    def test_sick(self, idx, model):
        r = search("아파서 누워있어 감기 열", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in HEALTH_IDX for x in r[:3])

    def test_yoga_english(self, idx, model):
        r = search("yoga stretching morning routine", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in HEALTH_IDX for x in r[:5])


# ===== ENTERTAINMENT (6) =====
class TestRealEntertainment:
    def test_kdrama(self, idx, model):
        r = search("넷플릭스 더글로리 드라마 시즌", idx, k=10, model=model)
        assert any(MESSAGES.index(x.message) in ENTERTAIN_IDX for x in r[:7])

    def test_webtoon(self, idx, model):
        r = search("웹툰 만화 추천", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in ENTERTAIN_IDX for x in r[:3])

    def test_kpop(self, idx, model):
        r = search("에스파 앨범 신곡 케이팝", idx, k=10, model=model)
        assert any(MESSAGES.index(x.message) in ENTERTAIN_IDX for x in r[:7])

    def test_movie_english(self, idx, model):
        r = search("movie film incredible", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in ENTERTAIN_IDX for x in r[:5])

    def test_youtube(self, idx, model):
        r = search("유튜브 알고리즘 영상", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in ENTERTAIN_IDX for x in r[:3])

    def test_weekend_activity(self, idx, model):
        r = search("주말에 뭐하지 놀거리", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in ENTERTAIN_IDX for x in r[:5])


# ===== SHOPPING (6) =====
class TestRealShopping:
    def test_delivery(self, idx, model):
        r = search("택배 배송 주문", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SHOPPING_IDX for x in r[:3])

    def test_headphones(self, idx, model):
        r = search("이어폰 헤드폰 음질", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SHOPPING_IDX for x in r[:5])

    def test_clothes_sale(self, idx, model):
        r = search("옷 할인 세일 쇼핑", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SHOPPING_IDX for x in r[:3])

    def test_phone_upgrade(self, idx, model):
        r = search("new phone upgrade worth it", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SHOPPING_IDX for x in r[:5])

    def test_credit_card(self, idx, model):
        r = search("카드값 지출 돈", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SHOPPING_IDX for x in r[:3])

    def test_secondhand(self, idx, model):
        r = search("중고거래 당근마켓 싸게", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SHOPPING_IDX for x in r[:3])


# ===== SOCIAL/RELATIONSHIP (6) =====
class TestRealSocial:
    def test_blind_date(self, idx, model):
        r = search("소개팅 만남 데이트", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SOCIAL_IDX for x in r[:3])

    def test_anniversary(self, idx, model):
        r = search("기념일 커플 연애", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SOCIAL_IDX for x in r[:3])

    def test_picnic(self, idx, model):
        r = search("친구 나들이 피크닉", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SOCIAL_IDX for x in r[:3])

    def test_ldr_english(self, idx, model):
        r = search("long distance relationship video call", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SOCIAL_IDX for x in r[:5])

    def test_reunion(self, idx, model):
        r = search("동창회 오랜만에 만남", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SOCIAL_IDX for x in r[:3])

    def test_breakup(self, idx, model):
        r = search("이별 헤어짐 슬퍼", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SOCIAL_IDX for x in r[:3])


# ===== CROSS-TOPIC HARD CASES (10) =====
class TestRealCrossTopic:
    def test_cross_en_to_kr_work(self, idx, model):
        """English query finds Korean work messages"""
        r = search("salary negotiation raise", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in WORK_IDX for x in r[:5])

    def test_cross_kr_to_en_health(self, idx, model):
        """Korean query finds English health messages"""
        r = search("요가 스트레칭 아침 운동 yoga", idx, k=10, model=model)
        assert any(MESSAGES.index(x.message) in HEALTH_IDX for x in r[:7])

    def test_slang_ㅋㅋ(self, idx, model):
        """슬랭/줄임말 검색"""
        r = search("웃긴거 ㅋㅋ 재미", idx, k=5, model=model)
        assert len(r) > 0

    def test_emoticon_search(self, idx, model):
        """이모티콘 포함 검색"""
        r = search("슬퍼 ㅠㅠ 힘들어", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SOCIAL_IDX for x in r[:5])

    def test_vague_who_said(self, idx, model):
        """모호한 검색: '누가 ~했던거'"""
        r = search("누가 할인이라고 했던거", idx, k=5, model=model)
        assert any(MESSAGES.index(x.message) in SHOPPING_IDX for x in r[:5])

    def test_topic_separation_work_vs_health(self, idx, model):
        """Work results score higher than health for work query"""
        r = search("회사 업무 프로젝트 마감", idx, k=30, model=model)
        work_scores = [x.score for x in r if MESSAGES.index(x.message) in WORK_IDX]
        health_scores = [x.score for x in r if MESSAGES.index(x.message) in HEALTH_IDX]
        if work_scores and health_scores:
            assert max(work_scores) > max(health_scores)

    def test_topic_separation_shopping_vs_social(self, idx, model):
        """Shopping results score higher than social for shopping query"""
        r = search("쇼핑 구매 결제 물건", idx, k=30, model=model)
        shop_scores = [x.score for x in r if MESSAGES.index(x.message) in SHOPPING_IDX]
        social_scores = [x.score for x in r if MESSAGES.index(x.message) in SOCIAL_IDX]
        if shop_scores and social_scores:
            assert max(shop_scores) > max(social_scores)

    def test_filter_room_with_real_search(self, idx, model):
        r = search("추천", idx, k=10, model=model, room="취업")
        assert all("취업" in x.message.room for x in r)

    def test_filter_date_with_real_search(self, idx, model):
        r = search("뭔가", idx, k=10, model=model,
                    after=datetime(2024, 2, 3), before=datetime(2024, 2, 5))
        for x in r:
            assert x.message.timestamp >= datetime(2024, 2, 3)
            assert x.message.timestamp < datetime(2024, 2, 5)

    def test_all_results_have_valid_fields(self, idx, model):
        r = search("아무거나", idx, k=10, model=model)
        for x in r:
            assert x.message.sender != ""
            assert x.message.content != ""
            assert x.message.platform == "discord"
            assert isinstance(x.score, float)
            assert isinstance(x.rank, int)

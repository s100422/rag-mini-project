"""본문 띄어쓰기 정리 (pymupdf4llm 본문의 줄바꿈 자리·글자 간격 문제).

문제 유형
- 공백 없는 긴 한글(`전세관련보증의경우필요시…`): PDF 글자 사이 간격이 없어 단어가 붙음 + 줄바꿈 자리에 엉뚱한 공백
- 줄바꿈 자리에서 단어가 갈라짐(`연 체율`, `적립금 은`, `1 년만 에`)
- 숫자와 단위 사이, 괄호 안쪽, 쉼표 앞의 불필요한 공백(`2025 년말`, `( 이하 ‘DB’)`, `하고 ,`)

줄(문단)마다 두 경로 중 하나를 쓴다. 텍스트·각주 블록에만 적용하고 표·비전 결과는 건드리지 않는다.
A. 공백 없는 줄(한글 10자 이상 이어진 덩어리가 있는 줄): 기존 공백은 믿을 수 없으므로 모두 지우고 Kiwi가 줄 전체를 다시 띄운다.
   Kiwi가 숫자·기호 주변에 넣는 어색한 공백(`315.6 조 원`, `75.9 %`)은 정리한다.
B. 정상 줄: 갈라진 말만 잇는다.
   - 숫자 뒤 단위가 따로 떨어져 있으면 붙인다(`2025 년말`, `431.7 조원`)
   - 조사·어미·접미사가 따로 떨어져 있으면 앞말에 붙인다(`적립금 은`, `있 다`) — 붙인 글자열을 Kiwi로 분석해 판정
   - 말 한가운데 갈라진 조각(`금 융`, `연 체율`)은 붙인 형태가 **같은 문서 다른 곳에 실제로 있을 때만** 붙인다
   - 괄호 안쪽·쉼표·마침표 앞 공백을 지운다

안전장치: 공백을 뺀 글자가 처리 전후로 같아야 하고, 아니면 그 줄은 원래대로 둔다.
"""
import re
from functools import lru_cache

from kiwipiepy import Kiwi

LONG_RUN_RE = re.compile(r"[가-힣]{10,}")          # 이 길이 이상 공백 없이 이어지면 "공백 없는 줄"(정상 문장의 한 어절은 보통 9자 이하)
UNIT_TOKEN_RE = re.compile(
    r"^(?:조원|억원|만원|천원|원|년|월|일|%p|%|개월|개|명|좌|배|세|분기|건|회)"
    r"(?:말|초|간|째|동안|이|가|은|는|의|에|를|을|로|으로|와|과|도|만|까지|부터|보다|대비|이상|이하|미만|초과|전|후|중|씩|에서|이다|입니다|처럼)?"
    r"[.,;:)\]]*$")
DETERMINERS = {"이", "그", "저", "각", "한", "두", "본", "당", "동", "현", "전", "타", "제"}   # 뒤 단어와 띄어 쓰는 지시어(뒤쪽에 올 때)
NO_GLUE_A = {"대"}                       # 숫자 뒤에 붙는 '…%대 낮은'의 대
# 홀로 선 한 글자 단어: 뒤 단어와 합치면 안 된다('수 있다', '중 연체'). 조사·어미가 따라올 때만 붙인다('것 으로').
STANDALONE_1 = set("수것등및중외때데바점곳편후간내상하약총각매또더안못잘")
# 앞말에 붙여 쓰는 한 글자 조사: 뒤 단어가 조사·어미 조각일 때만 합친다('에 서'). 뒤가 낱말이면 앞말의 조사이므로 건드리지 않는다('가 가장').
JOSA1 = set("가이은는을를의에도로와과만서며고다나")
ATTACH_TAGS = ("J", "E", "XS", "VCP")    # 조사·어미·접미사·서술격 조사: 앞말에 붙여 쓴다
PREFIX_RE = re.compile(r"^(?:#{1,6}\s+|[-*]\s+|>\s*|\[\^\d+\]:\s*)*")


class SpacingFixer:
    def __init__(self) -> None:
        self.kiwi = Kiwi()
        self._evidence = ""
        self.changes: list[tuple[str, str, str]] = []   # (종류, 전, 후) — 사람이 검토할 수 있게 남긴다

    def set_evidence(self, bodies: list[str]) -> None:
        """같은 문서에서 공백 없이 쓰인 한글 덩어리를 모은다(갈라진 조각을 붙일지 판단하는 근거)."""
        runs = set()
        for b in bodies:
            for tok in b.split():
                runs.update(re.findall(r"[가-힣]{2,}", tok))
        self._evidence = " ".join(runs)
        self._ev.cache_clear()

    @lru_cache(maxsize=None)
    def _ev(self, glued: str) -> bool:
        return glued in self._evidence

    def fix(self, body: str) -> str:
        return "\n".join(self.fix_line(ln) for ln in body.split("\n"))

    def fix_line(self, line: str) -> str:
        if not line.strip() or line.startswith(("|", "<!--", "```")):
            return line
        prefix = PREFIX_RE.match(line).group(0)
        content = line[len(prefix):]
        # 장 제목 배너('1재무설계의의의')는 글자 간격이 없는 경우가 많아 제목 줄은 6자 덩어리부터 다시 띄운다
        thr = 6 if prefix.lstrip().startswith("#") else 10
        if re.search(rf"[가-힣]{{{thr},}}", content):
            new = self._respace(content)
        else:
            new = self._fix_content(content)
        if re.sub(r"\s", "", new) != re.sub(r"\s", "", content):   # 글자가 바뀌었으면 되돌린다
            return line
        return prefix + new

    # ---- A. 공백 없는 줄 ---------------------------------------------------
    @lru_cache(maxsize=None)
    def _respace(self, content: str) -> str:
        compact = re.sub(r"\s+", "", content)
        spaced = self.kiwi.space(compact, reset_whitespace=True)
        self.changes.append(("respace", content[:60], spaced[:60]))
        return self._cleanup(spaced)

    # ---- B. 정상 줄 --------------------------------------------------------
    @lru_cache(maxsize=None)
    def _should_glue(self, a: str, b: str, after_digit: bool = False) -> bool:
        """a와 b 사이 공백을 지울지.

        - 뒤 형태소가 조사·어미·접미사·서술격 '이'면 앞말에 붙여 쓰므로 붙인다('적립금 은', '있 다', '것 입니다')
        - 공백이 형태소 한가운데이고 한쪽이 2자 이하인 조각이며, 붙인 형태가 같은 문서에 실제로 있으면 붙인다('금 융', '연 체율')
        - 그 밖에는 띄운 말이므로 그대로 둔다
        """
        if b in DETERMINERS:
            return False
        cut = len(a)
        toks = self.kiwi.tokenize(a + b)
        after = next((t for t in toks if t.start == cut), None)
        attach = after is not None and after.tag.startswith(ATTACH_TAGS)
        if a in JOSA1:                       # 앞말의 조사: 뒤가 조사·어미 조각일 때만
            return attach and len(b) <= 2
        if not any(t.start + t.len == cut for t in toks):
            # 증거: 붙인 앞부분(조각 + 뒤 한두 글자)이 같은 문서에 있어야 한다. 뒤 조사까지 포함하면 증거가 사라지므로 b는 앞부분만 쓴다.
            key = a + (b[:1] if len(a) >= 2 else b[:2])
            return (not after_digit and a not in STANDALONE_1 and min(len(a), len(b)) <= 2 and self._ev(key))
        return attach

    def _fix_content(self, s: str) -> str:
        tokens = s.split()
        i = 0
        while i + 1 < len(tokens):
            a, b = tokens[i], tokens[i + 1]
            if re.search(r"\d$", a) and UNIT_TOKEN_RE.match(b):
                self.changes.append(("unit", f"{a} {b}", a + b))
                tokens[i:i + 2] = [a + b]
                continue
            # 앞 토큰 끝의 굵은 글씨 표시('**보험약관**')는 떼고 글자를 본다
            ta, mb = re.search(r"[가-힣]+$", a.rstrip("*_")), re.match(r"^([가-힣]+)(.*)$", b)
            # 숫자 바로 뒤 단위('10일 이상')의 '경계 없음' 붙이기는 하지 않는다(조사·어미 붙이기는 한다: '20.1조원 으로')
            if ta and mb and ta.group(0) not in NO_GLUE_A \
                    and self._should_glue(ta.group(0), mb.group(1), bool(re.search(r"\d$", a[:ta.start()]))):
                self.changes.append(("glue", f"{ta.group(0)} {mb.group(1)}", ta.group(0) + mb.group(1)))
                tokens[i:i + 2] = [a + b]
                continue
            i += 1
        return self._cleanup(" ".join(tokens))

    @staticmethod
    def _cleanup(text: str) -> str:
        """괄호 안쪽·쉼표·마침표 앞 공백, 숫자와 단위·기호 사이 공백을 정리한다."""
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r"\s+([,，;])", r"\1", text)
        text = re.sub(r"(?<=[가-힣\d%)\]])\s+\.(?=\s|$)", ".", text)
        text = re.sub(r"(\d)\s+(조|억|만|천)\s*(원)", r"\1\2\3", text)          # 315.6 조 원 → 315.6조원
        text = re.sub(r"(\d)\s+(%p|%)", r"\1\2", text)                           # 75.9 % → 75.9%
        text = re.sub(r"·\s+", "·", text)                                        # 직· 간접적 → 직·간접적
        text = re.sub(r"\s+(\^\d{1,3}\))", r"\1", text)                          # 등 ^2) → 등^2)
        text = re.sub(r"(?<=[가-힣\d%])\s+\*", "*", text)                        # 증가 *하였다 → 증가*하였다
        return text

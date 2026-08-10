"""
Helix Tone AI Config Module
===========================
이 모듈은 오디오 전처리, Helix Native 이펙터 체인 스키마 정의,
딥러닝 모델 하이퍼파라미터 및 학습 설정을 관리합니다.
"""

from typing import Dict, List, Any
from collections import defaultdict

# ==========================================
# 1. 오디오 신호 처리 (Audio Processing) 설정
# ==========================================
SAMPLE_RATE: int = 44100        # 샘플링 레이트 (Hz)
DURATION: float = 5.0           # 오디오 길이 (초)
NUM_SAMPLES: int = int(SAMPLE_RATE * DURATION)

# Mel-Spectrogram 설정
N_FFT: int = 2048
HOP_LENGTH: int = 512
N_MELS: int = 128
F_MIN: float = 20.0
F_MAX: float = 20000.0


# ==========================================
# 2. Helix Native 이펙터 체인 스키마 정의
# ==========================================
# 모델 카테고리를 확장하여 Helix의 전체 이펙터 체인 슬롯 커버
SLOT_NAMES: List[str] = [
    "Distortion", "Dynamics", "EQ", "Modulation", "Delay", 
    "Reverb", "Pitch_Synth", "Filter", "Wah", "Volume_Pan", 
    "Amp", "Cab"
]
NUM_SLOTS: int = len(SLOT_NAMES)
MAX_KNOBS_PER_SLOT: int = 10

# PDF 매뉴얼을 참조한 전체 카테고리별 모델 카탈로그 (0번 index는 'None/Bypass')
MODEL_CATALOG: Dict[str, List[str]] = {
    "Distortion": [
        "None", "Compulsive Drive", "Dhyana Drive", "Horizon Drive", "Valve Driver", 
        "Top Secret OD", "Prize Drive", "Scream 808", "Pillars", "Hedgehog D9", 
        "Stupor OD", "Deez One Vintage", "Deez One Mod", "Ratatoullie Dist", "Vermin Dist", 
        "Vital Dist", "Vital Boost", "KWB", "Legendary Drive", "Swedish Chainsaw", 
        "Arbitrator Fuzz", "Pocket Fuzz", "Bighorn Fuzz", "Triangle Fuzz", "Dark Dove Fuzz", 
        "Ballistic Fuzz", "Industrial Fuzz", "Tycoctavia Fuzz", "Wringer Fuzz", "Thrifter Fuzz", 
        "Xenomorph Fuzz", "Megaphone", "Kinky Boost", "Deranged Master", "Minotaur", 
        "Teemah!", "Heir Apparent", "Tone Sovereign", "Alpaca Rogue", "Bitcrusher", 
        "Ampeg Scrambler", "ZeroAmp Bass DI", "Regal Bass DI", "Obsidian 7000", "Clawthorn Drive", 
    ],
    "Dynamics": [
        "None", "Deluxe Comp", "Red Squeeze", "Kinky Comp", "Ampeg Opto Comp", 
        "Rochester Comp", "LA Studio Comp", "3-Band Comp", "Noise Gate", "Hard Gate", 
        "Horizon Gate", "Autoswell", "Feedbacker"
    ],
    "EQ": [
        "None", "Simple EQ", "Low and High Cut", "Low/High Shelf", "Parametric", 
        "Tilt", "10 Band Graphic", "Cali Q Graphic", "Acoustic Sim"
    ],
    "Modulation": [
        "None", "Optical Trem", "60s Bias Trem", "Tremolo/Autopan", 
        "Harmonic Tremolo", "Bleat Chop Trem", "Script Mod Phase", "Pebble Phaser", 
        "Ubiquitous Vibe", "FlexoVibe", "Deluxe Phaser", "Gray Flanger", "Harmonic Flanger", 
        "Courtesan Flange", "Dynamix Flanger", "Chorus", "70s Chorus", "PlastiChorus", 
        "Ampeg Liquifier Chorus", "4-Voice Chorus", "Bubble Vibrato", 
        "Retro Reel", 
        "Double Take", "Poly Detune", "AM Ring Mod"
    ],
    "Delay": [
        "None", "Simple Delay", 
        "Mod/Chorus Echo",
        "Ducked Delay", "Reverse Delay", "Vintage Digital", "Vintage Swell", "Pitch Echo", 
        "Transistor Tape", "Cosmos Echo", "Bucket Brigade", "Adriatic Delay", 
        "Adriatic Swell", "Elephant Man", "Multi Pass", "Heliosphere", "Poly Sustain", 
        "Glitch Delay", "Euclidean Delay", "ADT", "Crisscross", "Tesselator", "Ratchet"
    ],
    "Reverb": [
        "None", "Dynamic Hall", "Dynamic Plate", "Dynamic Room", "Dynamic Ambience", 
        "Dynamic Bloom", "Shimmer", "Hot Springs", "Nonlinear", "Glitz", "Ganymede", 
        "Searchlights", "Plateaux", "Double Tank"
    ],
    "Pitch_Synth": [
        "None", "Pitch Wham", "Twin Harmony", "Simple Pitch", "Dual Pitch", "Boctaver", 
        "Poly Pitch", "Poly Wham", "Poly Capo", "12 String", "3 Note Generator", 
        "4 OSC Generator"
    ],
    "Filter": [
        "None", "Mutant Filter", "Mystery Filter", "Autofilter", "Asheville Pattrn"
    ],
    "Wah": [
        "None", "UK Wah 846", "Teardrop 310", "Fassel", "Weeper", "Chrome", "Chrome Custom", 
        "Throaty", "Vetta Wah", "Colorful", "Conductor", "Teardrop Bass Q"
    ],
    "Volume_Pan": [
        "None", "Volume Pedal", "Gain", "Pan", "Stereo Width", "Stereo Imager"
    ],
    "Amp": [
        "None", "US Deluxe Nrm", "US Deluxe Vib", "US Double Nrm", "US Double Vib", 
        "Mail Order Twin", "Divided Duo", "Interstate Zed", "Derailed Ingrid", "Grammatico GSG", 
        "Jazz Rivet 120", "Essex A15", "Essex A30", "A30 Fawn Nrm", "A30 Fawn Brt", 
        "Matchstick Ch1", "Matchstick Ch2", "Matchstick Jump", "Mandarin 80", "Mandarin Rocker", 
        "MOO)))N T Nrm", "MOO)))N T Brt", "MOO)))N T Jump", "Brit J45 Nrm", "Brit J45 Brt", 
        "Brit Trem Nrm", "Brit Trem Brt", "Brit Trem Jump", "Brit Plexi Nrm", "Brit Plexi Brt", 
        "Brit Plexi Jump", "Brit P75 Nrm", "Brit P75 Brt", "WhoWatt 100", "Soup Pro", 
        "Stone Age 185", "Voltage Queen", "Tweed Blues Nrm", "Tweed Blues Brt", "Fullerton Nrm", 
        "Fullerton Brt", "Fullerton Jump", "GrammaticoLG Nrm", "GrammaticoLG Brt", "GrammaticoLG Jmp", 
        "US Small Tweed", "US Princess", "US Super Nrm", "US Super Vib", "Brit 2203", "Brit 2204", 
        "Placater Clean", "Placater Dirty", "Cartographer", "German Xtra Blue", "German Xtra Red", 
        "German Mahadeva", "German Ubersonic", "Cali Texas Ch1", "Cali Texas Ch2", "Cali IV Rhythm 1", 
        "Cali IV Rhythm 2", "Cali IV Lead", "Cali Rectifire", "Archetype Clean", "Archetype Lead", 
        "ANGL Meteor", "Solo Lead Clean", "Solo Lead Crunch", "Solo Lead OD", "EV Panama Blue", 
        "EV Panama Red", "PV Panama", "PV Vitriol Clean", "PV Vitriol Crunch", "PV Vitriol Lead", 
        "Revv Gen Purple", "Revv Gen Red", "Das Benzin Mega", "Das Benzin Lead", "Line 6 Clarity", 
        "Line 6 Aristocrat", "Line 6 Carillon", "Line 6 Voltage", "Line 6 Kinetic", "Line 6 Oblivion", 
        "Line 6 Ventoux", "Line 6 Elmsley", "Line 6 Elektrik", "Line 6 Doom", "Line 6 Epic", 
        "Line 6 2204 Mod", "Line 6 Fatality", "Line 6 Litigator", "Line 6 Badonk", "Ampeg B-15NF", 
        "Ampeg SVT Nrm", "Ampeg SVT Brt", "Ampeg SVT-4 PRO", "US Dripman Nrm", "Woody Blue", 
        "Agua Sledge", "Agua 51", "Mandarin Bass 200", "Cali Bass", "Cali 400 Ch1", "Cali 400 Ch2", 
        "G Cougar 800", "Del Sol 300", "Busy One Ch1", "Busy One Ch2", "Busy One Jump", "Studio Tube Pre"
    ],
    "Cab": [
        "None", "Soup Pro Ellipse", "1x8 Small Tweed", "1x10 US Princess", "1x12 Fullerton", 
        "1x12 Grammatico", "1x12 US Deluxe", "1x12 Open Cast", "1x12 Open Cream", "1x12 Cali EXT", 
        "1x12 Cali IV", "1x12 Blue Bell", "2x12 Blue Bell", "2x12 Silver Bell", "2x12 Match H30", 
        "2x12 Match G25", "2x12 Double C12N", "2x12 Interstate", "2x12 Jazz Rivet", "2x12 Mail C12Q", 
        "2x12 Mandarin 30", "4x10 Tweed P10R", "4x10 US Super", "4x12 WhoWatt", "4x12 Greenback20", 
        "4x12 Greenback25", "4x12 Greenback30", "4x12 1960A T75", "4x12 Blackback30", "4x12 Brit V30", 
        "4x12 Cali V30", "4x12 Mandarin EM", "4x12 MOO)))N T75", "4x12 Cartog Guv", "4x12 Cartog C90", 
        "4x12 Uber T75", "4x12 Uber V30", "4x12 XXL V30", "4x12 SoloLead EM", "1x12 Epicenter", 
        "1x15 Ampeg B-15", "2x15 Brute", "2x15 US Dripman", "4x10 Garden", "4x10 Ampeg Pro", 
        "6x10 Cali Power", "8x10 SVT AV", "1x12 Field Coil", "1x12 Celest 12H", "1x12 Lead 80", 
        "1x18 Del Sol", "1x18 Woody Blue", "4x10 Ampeg HLF", "8x10 Ampeg SVT E"
    ]
}

# Helix 내 이펙터 블록 타입 매핑 (@type 필드) - 임의 매핑 포함
BLOCK_TYPE_MAP: Dict[str, int] = {
    "Distortion": 0,
    "Amp": 1,
    "Cab": 3,
    "Modulation": 4,
    "Dynamics": 5,
    "EQ": 6,
    "Delay": 7,
    "Reverb": 7,  # Delay와 동일한 계열일 수 있음
    "Pitch_Synth": 8,
    "Filter": 9,
    "Wah": 10,
    "Volume_Pan": 11
}

DEFAULT_DRIVE_KNOBS = {
    "Gain": {"min": 0.0, "max": 1.0, "default": 0.5},
    "Tone": {"min": 0.0, "max": 1.0, "default": 0.5},
    "Level": {"min": 0.0, "max": 1.0, "default": 0.5},
}
DEFAULT_AMP_KNOBS = {
    "Drive": {"min": 0.0, "max": 1.0, "default": 0.5},
    "Bass": {"min": 0.0, "max": 1.0, "default": 0.5},
    "Mid": {"min": 0.0, "max": 1.0, "default": 0.5},
    "Treble": {"min": 0.0, "max": 1.0, "default": 0.5},
    "Presence": {"min": 0.0, "max": 1.0, "default": 0.5},
    "ChVol": {"min": 0.0, "max": 1.0, "default": 0.8},
    "Master": {"min": 0.0, "max": 1.0, "default": 0.8},
}
DEFAULT_CAB_KNOBS = {
    "Mic": {"min": 0.0, "max": 15.0, "default": 1.0},
    "Distance": {"min": 1.0, "max": 12.0, "default": 1.0},
    "Position": {"min": 0.0, "max": 1.0, "default": 0.5},
    "Angle": {"min": 0.0, "max": 45.0, "default": 0.0},
    "Low Cut": {"min": 19.0, "max": 500.0, "default": 80.0},
    "High Cut": {"min": 1000.0, "max": 20100.0, "default": 8000.0},
}
DEFAULT_DELAY_KNOBS = {
    "Time": {"min": 0.0, "max": 2.0, "default": 0.5},
    "Feedback": {"min": 0.0, "max": 1.0, "default": 0.4},
    "Mix": {"min": 0.0, "max": 1.0, "default": 0.3},
}
DEFAULT_REVERB_KNOBS = {
    "Decay": {"min": 0.0, "max": 20.0, "default": 3.0},
    "Predelay": {"min": 0.0, "max": 0.2, "default": 0.02},
    "Mix": {"min": 0.0, "max": 1.0, "default": 0.3},
}
DEFAULT_DYNAMICS_KNOBS = {
    "Threshold": {"min": -60.0, "max": 0.0, "default": -20.0},
    "Ratio": {"min": 1.0, "max": 20.0, "default": 4.0},
    "Gain": {"min": 0.0, "max": 1.0, "default": 0.5},
}
DEFAULT_EQ_KNOBS = {
    "Low": {"min": -12.0, "max": 12.0, "default": 0.0},
    "Mid": {"min": -12.0, "max": 12.0, "default": 0.0},
    "High": {"min": -12.0, "max": 12.0, "default": 0.0},
    "Level": {"min": -12.0, "max": 12.0, "default": 0.0},
}
DEFAULT_MODULATION_KNOBS = {
    "Speed": {"min": 0.0, "max": 1.0, "default": 0.5},
    "Depth": {"min": 0.0, "max": 1.0, "default": 0.5},
    "Mix": {"min": 0.0, "max": 1.0, "default": 0.5},
}
DEFAULT_PITCH_KNOBS = {
    "Pitch": {"min": -12.0, "max": 12.0, "default": 0.0},
    "Mix": {"min": 0.0, "max": 1.0, "default": 0.5},
    "Level": {"min": 0.0, "max": 1.0, "default": 0.5},
}
DEFAULT_FILTER_KNOBS = {
    "Frequency": {"min": 20.0, "max": 10000.0, "default": 1000.0},
    "Q": {"min": 0.1, "max": 10.0, "default": 1.0},
    "Gain": {"min": -12.0, "max": 12.0, "default": 0.0},
}
DEFAULT_WAH_KNOBS = {
    "Position": {"min": 0.0, "max": 1.0, "default": 0.5},
    "Mix": {"min": 0.0, "max": 1.0, "default": 1.0},
}
DEFAULT_VOLUME_PAN_KNOBS = {
    "Volume": {"min": 0.0, "max": 1.0, "default": 1.0},
    "Pan": {"min": -1.0, "max": 1.0, "default": 0.0},
}

DEFAULT_CATEGORY_KNOBS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Distortion": DEFAULT_DRIVE_KNOBS,
    "Amp": DEFAULT_AMP_KNOBS,
    "Cab": DEFAULT_CAB_KNOBS,
    "Delay": DEFAULT_DELAY_KNOBS,
    "Reverb": DEFAULT_REVERB_KNOBS,
    "Dynamics": DEFAULT_DYNAMICS_KNOBS,
    "EQ": DEFAULT_EQ_KNOBS,
    "Modulation": DEFAULT_MODULATION_KNOBS,
    "Pitch_Synth": DEFAULT_PITCH_KNOBS,
    "Filter": DEFAULT_FILTER_KNOBS,
    "Wah": DEFAULT_WAH_KNOBS,
    "Volume_Pan": DEFAULT_VOLUME_PAN_KNOBS,
}

# -------------------------------------------------------------
# 명시적으로 정의된 특정 모델별 노브 범위 및 파라미터 (De-normalization용)
# -------------------------------------------------------------
KNOWN_KNOB_SCHEMA: Dict[str, Dict[str, Dict[str, Any]]] = {
    "Scream 808": DEFAULT_DRIVE_KNOBS,
    "Minotaur": DEFAULT_DRIVE_KNOBS,
    "Kinky Boost": {"Drive": {"min": 0.0, "max": 1.0, "default": 0.3}, "Boost": {"min": 0.0, "max": 1.0, "default": 1.0}},
    "Ratatoullie Dist": DEFAULT_DRIVE_KNOBS,
    
    "US Deluxe Nrm": DEFAULT_AMP_KNOBS,
    "Placater Clean": DEFAULT_AMP_KNOBS,
    "Brit 2203": DEFAULT_AMP_KNOBS,
    "Cali Rectifire": DEFAULT_AMP_KNOBS,
    
    "4x12 Cali V30": DEFAULT_CAB_KNOBS,
    "1x12 US Deluxe": DEFAULT_CAB_KNOBS,
    "4x12 Greenback25": DEFAULT_CAB_KNOBS,
    
    "Simple Delay": DEFAULT_DELAY_KNOBS,
    "Elephant Man": DEFAULT_DELAY_KNOBS,
    "Ping Pong": DEFAULT_DELAY_KNOBS,
    
    "Dynamic Hall": DEFAULT_REVERB_KNOBS,
    "Dynamic Plate": DEFAULT_REVERB_KNOBS,
    "Cave": DEFAULT_REVERB_KNOBS,
}

# 모델명 -> 카테고리 역매핑 매핑 딕셔너리
MODEL_TO_CATEGORY: Dict[str, str] = {}
for _category_name, _model_list in MODEL_CATALOG.items():
    for _m in _model_list:
        if _m != "None":
            MODEL_TO_CATEGORY[_m] = _category_name


class KnobSchemaDict(dict):
    """
    KNOWN_KNOB_SCHEMA에 없는 Helix Native 모델도 카테고리별 기본 노브 스키마(DEFAULT_CATEGORY_KNOBS)를
    자동 Fallback으로 연동하여 모든 모델에 대해 노브 파라미터가 유효하게 작동하도록 하는 커스텀 딕셔너리 클래스
    """
    def _resolve(self, key: str) -> Dict[str, Dict[str, Any]]:
        if key in KNOWN_KNOB_SCHEMA:
            return KNOWN_KNOB_SCHEMA[key]
        category = MODEL_TO_CATEGORY.get(key)
        if category and category in DEFAULT_CATEGORY_KNOBS:
            return DEFAULT_CATEGORY_KNOBS[category]
        return {}

    def __getitem__(self, key: str) -> Dict[str, Dict[str, Any]]:
        res = self._resolve(key)
        if res or key == "None":
            return res
        return super().get(key, {})

    def get(self, key: str, default=None) -> Dict[str, Dict[str, Any]]:
        if key == "None":
            return {}
        res = self._resolve(key)
        if res:
            return res
        return default if default is not None else {}

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str) or key == "None":
            return False
        return key in KNOWN_KNOB_SCHEMA or key in MODEL_TO_CATEGORY


# 모든 Helix 모델을 지원하도록 KnobSchemaDict 인스턴스로 바인딩
KNOB_SCHEMA = KnobSchemaDict()



# ==========================================
# 3. 딥러닝 모델 & 학습 설정
# ==========================================
BATCH_SIZE: int = 16
LEARNING_RATE: float = 1e-3
NUM_EPOCHS: int = 100

# Multi-task Loss Weighting (분류 vs 회귀)
LAMBDA_CLS: float = 1.0
LAMBDA_REG: float = 10.0
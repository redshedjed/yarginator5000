"""Track names and note layouts for the Rock Band / Clone Hero style notes.mid that YARG reads.

Keep format facts here (not scattered through generators) so they're easy to check and fix.
"""
from __future__ import annotations

from enum import IntEnum


class Difficulty(IntEnum):
    EASY = 0
    MEDIUM = 1
    HARD = 2
    EXPERT = 3


# --- track names ------------------------------------------------------------------------------
DRUMS = "PART DRUMS"
GUITAR = "PART GUITAR"
RHYTHM = "PART RHYTHM"
BASS = "PART BASS"
KEYS = "PART KEYS"
PRO_KEYS = {Difficulty.EXPERT: "PART REAL_KEYS_X", Difficulty.HARD: "PART REAL_KEYS_H",
            Difficulty.MEDIUM: "PART REAL_KEYS_M", Difficulty.EASY: "PART REAL_KEYS_E"}
VOCALS = "PART VOCALS"
HARM = {1: "HARM1", 2: "HARM2", 3: "HARM3"}
EVENTS = "EVENTS"
VENUE = "VENUE"
BEAT = "BEAT"

# --- instruments a song can have ([chart] instruments in song.toml) ----------------------------
INSTRUMENTS: dict[str, tuple[str, ...]] = {
    "drums": (DRUMS,),                       # pro drums
    "guitar": (GUITAR,),
    "rhythm": (RHYTHM,),
    "bass": (BASS,),
    "keys": tuple(PRO_KEYS.values()) + ("PART KEYS_ANIM_RH",),  # pro keys (4 difficulties + right-hand animation)
    "keys5": (KEYS,),                        # 5-lane keys
    "vocals": (VOCALS,),                     # solo vocals
    "harmonies": tuple(HARM.values()),       # HARM1-3
}
ALWAYS_TRACKS = (BEAT, EVENTS, VENUE)        # every chart has these


def tracks_for(instruments) -> set[str] | None:
    """Chart track names for a song's instruments (plus the always-on tracks); None = no restriction."""
    if instruments is None:
        return None
    unknown = [i for i in instruments if i not in INSTRUMENTS]
    if unknown:
        raise ValueError(f"unknown instrument(s) {unknown}; choose from {', '.join(INSTRUMENTS)}")
    return set(ALWAYS_TRACKS).union(*(INSTRUMENTS[i] for i in instruments))


# --- shared markers ---------------------------------------------------------------------------
OVERDRIVE = 116
SOLO = 103
TREMOLO = 126
TRILL = 127

# --- 5-lane (guitar / rhythm / bass / keys) ---------------------------------------------------
# Lane 0..4 = green, red, yellow, blue, orange at BASE + lane. Forced HOPO = BASE+5, forced strum = BASE+6.
FIVE_LANE_BASE = {Difficulty.EXPERT: 96, Difficulty.HARD: 84, Difficulty.MEDIUM: 72, Difficulty.EASY: 60}

# --- drums ------------------------------------------------------------------------------------
# Lane 0 = kick, 1 = red (snare), 2 = yellow, 3 = blue, 4 = green at BASE + lane.
DRUM_BASE = FIVE_LANE_BASE
DRUM_KICK_2X = 95
# Pro drums: yellow/blue/green are cymbals unless covered by these tom markers.
DRUM_TOM_MARKER = {"yellow": 110, "blue": 111, "green": 112}
DRUM_FILL = range(120, 125)  # all five notes together mark a drum fill / activation

# --- pro keys ---------------------------------------------------------------------------------
PRO_KEYS_RANGE = (48, 72)  # C3..C5, 25 keys
PRO_KEYS_RANGE_SHIFTS = {0: 48, 2: 50, 4: 52, 5: 53, 7: 55, 9: 57}  # range-shift note -> lowest visible key

# --- vocals / harmonies -----------------------------------------------------------------------
VOCAL_RANGE = (36, 84)
VOCAL_PHRASE = 105
VOCAL_PHRASE_2 = 106
VOCAL_PERCUSSION = 96

# --- BEAT track -------------------------------------------------------------------------------
BEAT_DOWNBEAT = 12
BEAT_UPBEAT = 13

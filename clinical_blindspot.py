import json
import math
import io
import os
import queue
import re
import threading
import time
import argparse
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf  # Required dependency; used for optional debug recording.
from flask import Flask, jsonify, request
from faster_whisper import WhisperModel

try:
	import webrtcvad
	WEBRTCVAD_AVAILABLE = True
except Exception:
	webrtcvad = None
	WEBRTCVAD_AVAILABLE = False


# -----------------------------
# Configuration
# -----------------------------

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
FRAME_BYTES = FRAME_SIZE * 2  # int16 mono
SILENCE_STOP_SECONDS = 1.0
MAX_CHUNK_SECONDS = 25.0
VAD_MODE = 2
ENERGY_VAD_THRESHOLD = 0.012

SPEAKER_MODE = "auto"  # "auto", "alternate", or "manual"
MANUAL_SPEAKER = "PATIENT"  # used when SPEAKER_MODE == "manual"

DEBUG_SAVE_AUDIO = False
DEBUG_AUDIO_PATH = "last_chunk.wav"
SESSION_LOG_PATH = "clinical_insights_log.jsonl"
MIN_AVG_CONFIDENCE = 0.40
WHISPER_MODEL_SIZE = "base"
LOCAL_WHISPER_MODEL_DIR = os.path.join(
	os.path.dirname(os.path.abspath(__file__)),
	"models",
	"faster-whisper-base",
)


DOCTOR_CUES = [
	"i suggest",
	"you should",
	"we should",
	"take this",
	"prescription",
	"medicine",
	"report",
	"test",
	"screening",
	"follow up",
	"get done",
	"dose",
]

PATIENT_CUES = [
	"i have",
	"i am",
	"i feel",
	"my",
	"since",
	"pain",
	"ache",
	"stomach",
	"fever",
	"cough",
	"breathless",
	"dizziness",
	"tired",
	"weak",
	"because",
	"for weeks",
	"for days",
	"last night",
	"cannot",
]

DOCTOR_ACTION_CUES = [
	"prescribe",
	"prescription",
	"recommend",
	"refer",
	"order",
	"schedule",
	"review",
	"monitor",
	"adjust",
	"dose",
	"start",
	"stop",
	"continue",
	"follow-up",
	"follow up",
	"advise",
	"evaluate",
]

DOCTOR_TEST_TERMS = [
	"ecg",
	"troponin",
	"echocardiogram",
	"hba1c",
	"fasting glucose",
	"cbc",
	"crp",
	"culture",
	"blood pressure",
	"renal panel",
	"mri",
	"mri brain",
	"tsh",
	"t3",
	"t4",
	"chest x-ray",
	"spirometry",
	"neurological exam",
]

DOCTOR_MEDICATION_TERMS = [
	"medicine",
	"cold medicine",
	"paracetamol",
	"acetaminophen",
	"ibuprofen",
	"antihistamine",
	"cough syrup",
	"dextromethorphan",
	"naproxen",
	"antipyretic",
	"painkiller",
	"analgesic",
]

SYMPTOM_TREATMENT_MAP = {
	"fever": ["cold medicine", "paracetamol", "acetaminophen", "ibuprofen", "antipyretic"],
	"headache": ["paracetamol", "acetaminophen", "ibuprofen", "painkiller", "analgesic"],
	"stomach ache": ["antacid", "omeprazole", "h2 blocker", "antispasmodic"],
	"cough": ["cold medicine", "cough syrup", "antihistamine", "dextromethorphan"],
	"shortness of breath": ["inhaler", "nebulizer", "bronchodilator"],
	"chest pain": ["ecg", "troponin", "echocardiogram", "cardiac"],
	"fatigue": ["hba1c", "cbc", "thyroid", "tsh", "t3", "t4"],
	"weight loss": ["hba1c", "fasting glucose", "tsh", "t3", "t4"],
	"dizziness": ["cbc", "electrolytes", "blood pressure"],
	"vision problem": ["mri", "neurological exam", "ophthalmology"],
	"thirst": ["hba1c", "fasting glucose", "electrolytes"],
	"nausea": ["antacid", "antiemetic"],
	"anxiety": ["clinical assessment", "thyroid panel"],
}

FOLLOW_UP_QUESTIONS = {
	"fever": "Ask about onset, pattern, and any recent infections or exposures.",
	"headache": "Ask about headache quality, severity, location, and any visual changes.",
	"stomach ache": "Ask about pain location, changes with food, and any nausea or vomiting.",
	"cough": "Ask about cough duration, sputum, and whether it worsens at night.",
	"shortness of breath": "Ask about exertional triggers, orthopnea, and exercise tolerance.",
	"chest pain": "Ask about pain characteristics, radiation, and associated shortness of breath.",
	"fatigue": "Ask about sleep, appetite, weight changes, and daily function.",
	"weight loss": "Ask about appetite, diet changes, and unintended weight loss duration.",
	"dizziness": "Ask about dizziness timing, triggers, and associated chest pain or palpitations.",
	"vision problem": "Ask about sudden vision changes, blurring, or double vision.",
	"thirst": "Ask about fluid intake, urinary frequency, and recent dietary changes.",
}

COMMUNICATION_EMPATHY_PHRASES = [
	"i understand",
	"that sounds",
	"i'm sorry",
	"i am sorry",
	"thank you",
	"i can imagine",
	"i hear you",
	"that must be difficult",
	"take your time",
	"let's review",
]

COMMUNICATION_DIRECTIVE_PHRASES = [
	"you should",
	"you must",
	"take this",
	"you need",
	"don't",
	"do not",
	"you have to",
	"i want you to",
	"we'll do",
	"we should",
]

COMMUNICATION_DISMISSIVE_PHRASES = [
	"it's nothing",
	"it's fine",
	"just rest",
	"it's probably",
	"don't worry",
	"no need",
	"not serious",
]

COMMUNICATION_BIAS_PHRASES = [
	"you're exaggerating",
	"it's all in your head",
	"you're too sensitive",
	"you look young",
	"you don't seem",
	"you are healthy",
]


def count_phrase_matches(text: str, phrases: List[str]) -> List[str]:
	normalized = normalize_text(text)
	return [phrase for phrase in phrases if phrase in normalized]


def detect_communication_bias(patient_turns: List[str], doctor_turns: List[str]) -> Dict:
	patient_text = " ".join(patient_turns)
	doctor_text = " ".join(doctor_turns)
	doctor_words = len(normalize_text(doctor_text).split())
	patient_words = len(normalize_text(patient_text).split())
	ratio = round(doctor_words / max(1, patient_words), 2)

	empathy_matches = count_phrase_matches(doctor_text, COMMUNICATION_EMPATHY_PHRASES)
	directive_matches = count_phrase_matches(doctor_text, COMMUNICATION_DIRECTIVE_PHRASES)
	dismissive_matches = count_phrase_matches(doctor_text, COMMUNICATION_DISMISSIVE_PHRASES)
	bias_matches = count_phrase_matches(doctor_text, COMMUNICATION_BIAS_PHRASES)

	flags: List[str] = []
	if ratio >= 2.5:
		flags.append("doctor_dominant")
	elif ratio <= 0.4:
		flags.append("patient_dominant")

	if not empathy_matches and patient_words >= 10 and doctor_words > 0:
		flags.append("possible_empathy_gap")

	if len(directive_matches) >= 3:
		flags.append("directive_tone")

	if dismissive_matches:
		flags.append("dismissive_language")

	if bias_matches:
		flags.append("potential_bias_language")

	return {
		"doctor_word_count": doctor_words,
		"patient_word_count": patient_words,
		"doctor_patient_ratio": ratio,
		"empathy_phrases": empathy_matches,
		"directive_phrases": directive_matches,
		"dismissive_phrases": dismissive_matches,
		"bias_phrases": bias_matches,
		"flags": flags,
	}


# -----------------------------
# Layer 1: Audio Capture Layer
# -----------------------------

def _audio_callback(indata, frames, callback_time, status, audio_q):
	if status:
		print(f"[Audio warning] {status}")
	audio_q.put(bytes(indata))


def capture_speech_chunk(
	sample_rate: int = SAMPLE_RATE,
	silence_stop_seconds: float = SILENCE_STOP_SECONDS,
	vad_mode: int = VAD_MODE,
	max_chunk_seconds: float = MAX_CHUNK_SECONDS,
) -> Optional[np.ndarray]:
	"""
	Continuously listens and captures a single speech chunk.
	Recording starts when VAD detects voice and stops after ~1s silence.
	Returns float32 audio in range [-1, 1] or None if no speech was detected.
	"""
	vad = webrtcvad.Vad(vad_mode) if WEBRTCVAD_AVAILABLE else None
	audio_q: queue.Queue = queue.Queue()
	frame_duration_s = FRAME_DURATION_MS / 1000.0

	speech_frames: List[bytes] = []
	speech_started = False
	silence_duration = 0.0
	started_at = time.time()

	callback = lambda indata, frames, callback_time, status: _audio_callback(
		indata, frames, callback_time, status, audio_q
	)

	with sd.RawInputStream(
		samplerate=sample_rate,
		blocksize=FRAME_SIZE,
		channels=CHANNELS,
		dtype="int16",
		callback=callback,
	):
		while True:
			if time.time() - started_at > max_chunk_seconds:
				break

			try:
				frame = audio_q.get(timeout=0.5)
			except queue.Empty:
				continue

			if len(frame) < FRAME_BYTES:
				continue

			if vad is not None:
				is_speech = vad.is_speech(frame, sample_rate)
			else:
				# Fallback VAD for environments where webrtcvad wheel is unavailable.
				frame_i16 = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
				rms = float(np.sqrt(np.mean(np.square(frame_i16))) / 32768.0)
				is_speech = rms >= ENERGY_VAD_THRESHOLD

			if is_speech:
				speech_started = True
				silence_duration = 0.0
				speech_frames.append(frame)
			elif speech_started:
				speech_frames.append(frame)
				silence_duration += frame_duration_s
				if silence_duration >= silence_stop_seconds:
					break

	if not speech_frames:
		return None

	pcm = b"".join(speech_frames)
	audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
	return audio


# -----------------------------
# Layer 2: Speech-to-Text Layer
# -----------------------------

def init_whisper_model() -> Tuple[WhisperModel, str]:
	"""Initialize faster-whisper model, preferring GPU when available."""
	model_source = LOCAL_WHISPER_MODEL_DIR if os.path.isdir(LOCAL_WHISPER_MODEL_DIR) else WHISPER_MODEL_SIZE

	try:
		return WhisperModel(model_source, device="cuda", compute_type="float16"), "cuda"
	except Exception:
		try:
			return WhisperModel(model_source, device="cpu", compute_type="int8"), "cpu"
		except Exception as e:
			err = str(e)
			if (
				"LocalEntryNotFoundError" in err
				or "ConnectError" in err
				or "getaddrinfo failed" in err
			):
				raise RuntimeError(
					"Whisper model is not available locally and download failed. "
					"Use setup_whisper_model.bat on a machine with internet, "
					"or copy the folder models/faster-whisper-base into this project."
				) from e
			raise


def transcribe_audio_chunk(model: WhisperModel, audio: np.ndarray) -> Dict:
	"""
	Transcribe audio with faster-whisper and return structured output.
	"""
	segments, _ = model.transcribe(
		audio,
		beam_size=1,
		language="en",
		vad_filter=False,
	)

	segment_dicts = []
	full_text_parts = []
	for idx, seg in enumerate(segments):
		text = seg.text.strip()
		full_text_parts.append(text)
		confidence = max(0.0, min(1.0, math.exp(getattr(seg, "avg_logprob", -3.0))))
		segment_dicts.append(
			{
				"id": idx,
				"start": float(seg.start),
				"end": float(seg.end),
				"text": text,
				"confidence": round(confidence, 3),
			}
		)

	return {
		"segments": segment_dicts,
		"full_text": " ".join([t for t in full_text_parts if t]).strip(),
	}


def average_confidence(transcription: Dict) -> float:
	segments = transcription.get("segments", [])
	if not segments:
		return 0.0
	return sum(seg.get("confidence", 0.0) for seg in segments) / len(segments)


# -----------------------------
# Layer 3: Transcript Structuring Layer
# -----------------------------

def infer_speaker_from_text(text: str, previous_speaker: Optional[str]) -> str:
	normalized = normalize_text(text)
	doctor_terms = DOCTOR_CUES + DOCTOR_ACTION_CUES + DOCTOR_MEDICATION_TERMS + DOCTOR_TEST_TERMS
	doctor_score = sum(1 for cue in doctor_terms if cue in normalized)
	patient_score = sum(1 for cue in PATIENT_CUES if cue in normalized)

	if doctor_score > patient_score:
		return "DOCTOR"
	if patient_score > doctor_score:
		return "PATIENT"

	if previous_speaker == "DOCTOR":
		return "PATIENT"
	if previous_speaker == "PATIENT":
		return "DOCTOR"

	return "PATIENT"


def assign_speaker_label(
	chunk_index: int,
	text: str,
	session_state: Dict,
	mode: str = SPEAKER_MODE,
	manual_speaker: str = MANUAL_SPEAKER,
) -> str:
	"""
	Auto: infer from language cues, fallback to turn-taking.
	Alternate: deterministic PATIENT/DOCTOR by chunk index.
	Manual: fixed label from MANUAL_SPEAKER.
	"""
	if mode == "manual":
		return manual_speaker.upper()
	if mode == "auto":
		return infer_speaker_from_text(text, session_state.get("last_speaker"))
	return "PATIENT" if chunk_index % 2 == 0 else "DOCTOR"


def structure_transcript(transcription: Dict, speaker_label: str) -> Dict:
	structured_segments = []
	for seg in transcription["segments"]:
		item = dict(seg)
		item["speaker"] = speaker_label
		structured_segments.append(item)

	return {
		"segments": structured_segments,
		"full_text": transcription.get("full_text", ""),
		"speaker": speaker_label,
	}


# -----------------------------
# Layer 4: NLP Symptom Extraction Layer
# -----------------------------

SYMPTOM_SYNONYMS: Dict[str, List[str]] = {
	"fatigue": ["tired", "weak", "low energy", "fatigue"],
	"weight loss": ["lost weight", "losing weight", "weight loss"],
	"chest pain": ["chest discomfort", "tightness", "chest pain"],
	"shortness of breath": ["breathless", "difficulty breathing", "shortness of breath"],
	"thirst": ["very thirsty", "increased thirst", "thirst"],
	"headache": ["head pain", "migraine", "headache"],
	"stomach ache": ["stomach ache", "stomach pain", "abdominal pain", "tummy ache"],
	"vision problem": ["blurred vision", "double vision", "vision problem"],
	"fever": ["high temperature", "temperature", "fever"],
	"cough": ["coughing", "dry cough", "cough"],
	"dizziness": ["lightheaded", "faint", "dizziness"],
	"palpitations": ["racing heart", "heart pounding", "palpitations"],
	"nausea": ["nauseous", "queasy", "vomit", "nausea"],
	"anxiety": ["anxious", "nervous", "panic attack", "panic"],
}

REGEX_PATTERNS: Dict[str, str] = {
	"weight loss": r"\b(lost|losing)\s+weight\b",
	"shortness of breath": r"\b(shortness\s+of\s+breath|difficulty\s+breathing|breathless)\b",
	"chest pain": r"\b(chest\s+pain|chest\s+discomfort|tightness\s+in\s+chest)\b",
	"vision problem": r"\b(blurred\s+vision|double\s+vision|vision\s+problem)\b",
}


def normalize_text(text: str) -> str:
	text = text.lower()
	text = re.sub(r"[^a-z0-9\s]", " ", text)
	text = re.sub(r"\s+", " ", text).strip()
	return text


def extract_symptoms(text: str) -> List[str]:
	"""
	Hybrid extraction using keyword matching, synonym normalization, and regex.
	Returns unique canonical symptom list.
	"""
	normalized = normalize_text(text)
	found: Set[str] = set()

	for canonical, phrases in SYMPTOM_SYNONYMS.items():
		for phrase in phrases:
			if phrase in normalized:
				found.add(canonical)
				break

	for canonical, pattern in REGEX_PATTERNS.items():
		if re.search(pattern, normalized):
			found.add(canonical)

	return sorted(found)


def extract_doctor_actions(text: str) -> Set[str]:
	normalized = normalize_text(text)
	actions: List[str] = []
	doctor_terms = sorted(
		DOCTOR_TEST_TERMS + DOCTOR_MEDICATION_TERMS + DOCTOR_ACTION_CUES,
		key=lambda term: -len(term),
	)

	for term in doctor_terms:
		pattern = r"\b" + re.escape(term) + r"\b"
		for match in re.finditer(pattern, normalized):
			start, end = match.span()
			if any(start >= s and end <= e for s, e in [re.search(r"\b" + re.escape(t) + r"\b", normalized).span() for t in actions]):
				continue
			if term not in actions:
				actions.append(term)

	return set(actions)


def split_into_sentences(text: str) -> List[str]:
	if not text:
		return []
	parts = re.split(r'(?<=[.!?])\s+', text.strip())
	return [p.strip() for p in parts if p.strip()]


def find_cue_boundaries(text: str, cues: List[str]) -> List[int]:
	lower_text = text.lower()
	matched_spans: List[Tuple[int, int]] = []

	for cue in sorted(cues, key=lambda c: -len(c)):
		pattern = r"\b" + re.escape(cue) + r"\b"
		for match in re.finditer(pattern, lower_text):
			start, end = match.start(), match.end()
			# ignore cues that are contained inside already matched longer phrases
			if any(start >= span_start and end <= span_end for span_start, span_end in matched_spans):
				continue
			matched_spans.append((start, end))

	matched_spans.sort()
	return [span_start for span_start, span_end in matched_spans]


def get_doctor_boundaries(text: str) -> List[int]:
	return find_cue_boundaries(text, DOCTOR_CUES + DOCTOR_MEDICATION_TERMS + DOCTOR_ACTION_CUES)


def split_text_by_boundaries(text: str, boundaries: List[int]) -> List[str]:
	if not boundaries:
		return [text.strip()] if text.strip() else []

	parts: List[str] = []
	current = 0
	for boundary in boundaries:
		if boundary <= current:
			continue
		chunk = text[current:boundary].strip()
		if chunk:
			parts.append(chunk)
		current = boundary

	last_chunk = text[current:].strip()
	if last_chunk:
		parts.append(last_chunk)

	return parts


def assign_speaker_turns(text: str, forced_speaker: Optional[str] = None) -> List[Dict[str, str]]:
	if forced_speaker in {"PATIENT", "DOCTOR"}:
		return [{"speaker": forced_speaker, "text": text.strip()}] if text.strip() else []

	text = text.strip()
	if not text:
		return []

	sentences = split_into_sentences(text)
	if len(sentences) > 1:
		turns: List[Dict[str, str]] = []
		last_speaker: Optional[str] = None
		for sentence in sentences:
			speaker = infer_speaker_from_text(sentence, last_speaker)
			turns.append({"speaker": speaker, "text": sentence})
			last_speaker = speaker
		return turns

	# One long sentence or free-form text: split using doctor cue boundaries.
	doctor_boundaries = get_doctor_boundaries(text)
	if doctor_boundaries:
		chunks = split_text_by_boundaries(text, doctor_boundaries)
		turns = []
		last_speaker: Optional[str] = None
		for chunk in chunks:
			speaker = infer_speaker_from_text(chunk, last_speaker)
			turns.append({"speaker": speaker, "text": chunk})
			last_speaker = speaker
		return turns

	speaker = infer_speaker_from_text(text, None)
	return [{"speaker": speaker, "text": text.strip()}]


def is_multi_party_text(text: str) -> bool:
	normalized = normalize_text(text)
	if len(split_into_sentences(text)) > 1:
		return True
	if get_doctor_boundaries(text):
		return True
	patient_present = any(cue in normalized for cue in PATIENT_CUES)
	doctor_present = any(
		term in normalized
		for term in DOCTOR_ACTION_CUES + DOCTOR_MEDICATION_TERMS + DOCTOR_TEST_TERMS
	)
	return patient_present and doctor_present


def extract_symptoms_from_turns(turns: List[Dict[str, str]]) -> List[str]:
	all_symptoms: Set[str] = set()
	for turn in turns:
		if turn["speaker"] == "PATIENT":
			symptoms = extract_symptoms(turn["text"])
			all_symptoms.update(symptoms)
	return sorted(all_symptoms)


def get_remedies_for_symptom(symptom: str, doctor_text: str) -> List[str]:
	remedies = []
	normalized = normalize_text(doctor_text)
	for remedy, targets in SYMPTOM_TREATMENT_MAP.items():
		if remedy in normalized:
			for target in targets:
				if target == symptom and remedy not in remedies:
					remedies.append(remedy)

	for remedy, targets in SYMPTOM_TREATMENT_MAP.items():
		if symptom in targets and remedy in normalized and remedy not in remedies:
			remedies.append(remedy)

	return remedies


def build_patient_checklist(patient_symptoms: List[str], doctor_text: str) -> List[Dict[str, object]]:
	checklist: List[Dict[str, object]] = []
	normalized_doctor = normalize_text(doctor_text)
	for symptom in sorted(set(patient_symptoms)):
		symptom_remedies = SYMPTOM_TREATMENT_MAP.get(symptom, [])
		addressed = False
		related_remedies: List[str] = []

		for remedy in symptom_remedies:
			if remedy in normalized_doctor:
				addressed = True
				if remedy not in related_remedies:
					related_remedies.append(remedy)

		checklist.append({
			"symptom": symptom,
			"addressed": addressed,
			"related_remedies": related_remedies,
		})

	return checklist


def build_doctor_checklist(doctor_turns: List[str]) -> List[Dict[str, object]]:
	checklist: List[Dict[str, object]] = []
	for text in doctor_turns:
		checklist.append({
			"text": text,
			"actions": sorted(extract_doctor_actions(text)),
		})
	return checklist


def build_follow_up_questions(uncovered_symptoms: List[str], disease_matches: List[Dict]) -> List[str]:
	questions: List[str] = []
	for symptom in uncovered_symptoms:
		questions.append(FOLLOW_UP_QUESTIONS.get(symptom, f"Ask more about {symptom}."))

	if not uncovered_symptoms:
		questions.append("Confirm response to current treatment and monitor symptom resolution.")

	for item in disease_matches[:3]:
		questions.append(f"Consider evaluation for {item['disease']} (possible tests: {', '.join(item.get('tests', []))}).")

	return questions


def process_text_conversation(
	text: str,
	kb: List[Dict],
	session_state: Dict,
	forced_speaker: Optional[str] = None,
) -> Optional[Dict]:
	if not text.strip():
		return None

	turns = assign_speaker_turns(text, forced_speaker)
	if not turns:
		return None

	patient_turns: List[str] = []
	doctor_turns: List[str] = []
	for turn in turns:
		if turn["speaker"] == "PATIENT":
			patient_turns.append(turn["text"])
			update_session_memory(session_state, "PATIENT", turn["text"], extract_symptoms(turn["text"]))
		else:
			doctor_turns.append(turn["text"])
			update_session_memory(session_state, "DOCTOR", turn["text"], [])

	session_symptoms = sorted(session_state["all_symptoms"])
	disease_matches = match_diseases(session_symptoms, kb)
	blindspots = detect_blindspots(session_symptoms, disease_matches)
	score = compute_consultation_score(session_symptoms, disease_matches, kb)

	all_doctor_text = " ".join(doctor_turns)
	doctor_actions = sorted(extract_doctor_actions(all_doctor_text))
	patient_checklist = build_patient_checklist(session_symptoms, all_doctor_text)
	doctor_checklist = build_doctor_checklist(doctor_turns)
	covered = [item["symptom"] for item in patient_checklist if item["addressed"]]
	uncovered = [item["symptom"] for item in patient_checklist if not item["addressed"]]
	follow_up = build_follow_up_questions(uncovered, disease_matches)
	communication_bias = detect_communication_bias(patient_turns, doctor_turns)
	condition_checklist = build_condition_checklist(
		session_symptoms,
		disease_matches,
		kb,
		set(doctor_actions),
		doctor_turns,
	)

	result = build_final_output(
		transcription={
			"segments": [{"id": idx, "start": 0.0, "end": 0.0, "text": turn["text"], "confidence": 1.0} for idx, turn in enumerate(turns)],
			"full_text": text.strip(),
			"speaker": turns[-1]["speaker"],
		},
		symptoms_detected=session_symptoms,
		session_symptoms=session_symptoms,
		possible_conditions=disease_matches,
		missed_considerations=blindspots,
		consultation_score=score,
		patient_transcript=patient_turns,
		doctor_transcript=doctor_turns,
		doctor_actions=doctor_actions,
		condition_checklist=condition_checklist,
		unaddressed_conditions=[item["disease"] for item in condition_checklist if item["status"] != "addressed"],
		patient_checklist=patient_checklist,
		doctor_checklist=doctor_checklist,
		follow_up_questions=follow_up,
		communication_bias=communication_bias,
	)

	session_state["processed_chunks"] = session_state.get("processed_chunks", 0) + 1
	session_state["last_result"] = result
	return result


def build_condition_checklist(
	session_symptoms: List[str],
	disease_matches: List[Dict],
	kb: List[Dict],
	doctor_actions: Set[str],
	all_doctor_text: List[str],
) -> List[Dict]:
	checklist: List[Dict] = []
	full_doctor_text = normalize_text(" ".join(all_doctor_text))
	for item in disease_matches[:5]:
		disease_name = item["disease"]
		kb_item = next((entry for entry in kb if entry["disease"] == disease_name), None)
		if not kb_item:
			continue

		expected = set(kb_item["symptoms"])
		matched = sorted(expected.intersection(set(session_symptoms)))
		missing = sorted(expected.difference(set(session_symptoms)))
		recommended_tests = kb_item.get("tests", [])

		addressed = False
		if disease_name in full_doctor_text:
			addressed = True
		for test in recommended_tests:
			if normalize_text(test) in full_doctor_text:
				addressed = True
		for action in doctor_actions:
			if action in full_doctor_text:
				addressed = True

		status = "addressed" if addressed else "untouched"
		if matched and not missing and addressed:
			status = "addressed"
		elif matched and missing and addressed:
			status = "partial"
		elif matched and not missing and not addressed:
			status = "partial"

		checklist.append(
			{
				"disease": disease_name,
				"score": item.get("score", 0.0),
				"matched_symptoms": matched,
				"missing_symptoms": missing,
				"recommended_tests": recommended_tests,
				"status": status,
			}
		)

	return checklist


# -----------------------------
# Layer 5: Medical Knowledge Layer
# -----------------------------

def load_medical_knowledge_base() -> List[Dict]:
	"""Simplified in-code knowledge base inspired by public symptom-disease datasets."""
	return [
		{
			"disease": "diabetes",
			"symptoms": ["fatigue", "weight loss", "thirst", "vision problem"],
			"tests": ["HbA1c", "fasting glucose"],
			"weights": {"fatigue": 0.7, "weight loss": 0.8, "thirst": 0.9, "vision problem": 0.6},
		},
		{
			"disease": "anemia",
			"symptoms": ["fatigue", "dizziness", "shortness of breath"],
			"tests": ["CBC", "serum ferritin"],
			"weights": {"fatigue": 0.9, "dizziness": 0.7, "shortness of breath": 0.6},
		},
		{
			"disease": "heart disease",
			"symptoms": ["chest pain", "shortness of breath", "fatigue", "palpitations"],
			"tests": ["ECG", "troponin", "echocardiogram"],
			"weights": {"chest pain": 0.95, "shortness of breath": 0.85, "fatigue": 0.5, "palpitations": 0.7},
		},
		{
			"disease": "hypertension",
			"symptoms": ["headache", "vision problem", "dizziness"],
			"tests": ["blood pressure monitoring", "renal panel"],
			"weights": {"headache": 0.6, "vision problem": 0.7, "dizziness": 0.5},
		},
		{
			"disease": "infection",
			"symptoms": ["fever", "fatigue", "cough"],
			"tests": ["CBC", "CRP", "culture"],
			"weights": {"fever": 0.95, "fatigue": 0.5, "cough": 0.6},
		},
		{
			"disease": "neurological disorder",
			"symptoms": ["headache", "vision problem", "dizziness"],
			"tests": ["neurological exam", "MRI brain"],
			"weights": {"headache": 0.75, "vision problem": 0.75, "dizziness": 0.6},
		},
		{
			"disease": "thyroid disorder",
			"symptoms": ["fatigue", "weight loss", "palpitations"],
			"tests": ["TSH", "T3/T4"],
			"weights": {"fatigue": 0.7, "weight loss": 0.6, "palpitations": 0.75},
		},
		{
			"disease": "respiratory condition",
			"symptoms": ["cough", "shortness of breath", "chest pain"],
			"tests": ["chest X-ray", "spirometry"],
			"weights": {"cough": 0.7, "shortness of breath": 0.9, "chest pain": 0.6},
		},
		{
			"disease": "dehydration",
			"symptoms": ["thirst", "dizziness", "fatigue"],
			"tests": ["electrolytes", "BUN/creatinine"],
			"weights": {"thirst": 0.9, "dizziness": 0.7, "fatigue": 0.5},
		},
		{
			"disease": "anxiety disorder",
			"symptoms": ["shortness of breath", "chest pain", "palpitations"],
			"tests": ["clinical assessment", "thyroid panel"],
			"weights": {"shortness of breath": 0.5, "chest pain": 0.45, "palpitations": 0.8},
		},
	]


# -----------------------------
# Layer 6: Disease Matching Engine
# -----------------------------

def match_diseases(extracted_symptoms: List[str], kb: List[Dict]) -> List[Dict]:
	symptom_set = set(extracted_symptoms)
	matches = []

	for disease in kb:
		score = 0.0
		matched = []
		for symptom, weight in disease["weights"].items():
			if symptom in symptom_set:
				score += weight
				matched.append(symptom)

		if score > 0:
			matches.append(
				{
					"disease": disease["disease"],
					"score": round(score, 3),
					"tests": disease["tests"],
					"matched_symptoms": matched,
				}
			)

	matches.sort(key=lambda x: x["score"], reverse=True)
	return matches


# -----------------------------
# Layer 7: Blindspot Detection Engine
# -----------------------------

def detect_blindspots(extracted_symptoms: List[str], disease_matches: List[Dict]) -> Dict:
	symptom_set = set(extracted_symptoms)
	missed_questions: List[str] = []
	suggested_tests: Set[str] = set()
	risk_flags: List[str] = []

	if {"fatigue", "weight loss"}.issubset(symptom_set):
		missed_questions.append("Ask about appetite changes and consider diabetes screening.")
		suggested_tests.update(["HbA1c", "fasting glucose"])
		risk_flags.append("metabolic_red_flag")

	if "chest pain" in symptom_set:
		missed_questions.append("Ask about shortness of breath and exertional triggers.")
		suggested_tests.update(["ECG", "troponin"])
		risk_flags.append("cardiac_red_flag")

	if {"headache", "vision problem"}.issubset(symptom_set):
		missed_questions.append("Consider focused neurological evaluation.")
		suggested_tests.update(["neurological exam", "MRI brain"])
		risk_flags.append("neuro_red_flag")

	if {"fever", "fatigue"}.issubset(symptom_set):
		missed_questions.append("Screen for ongoing infection source and progression.")
		suggested_tests.update(["CBC", "CRP", "culture"])
		risk_flags.append("infection_red_flag")

	for item in disease_matches[:3]:
		for test in item.get("tests", []):
			suggested_tests.add(test)

	return {
		"missed_questions": missed_questions,
		"suggested_tests": sorted(suggested_tests),
		"risk_flags": sorted(set(risk_flags)),
	}


# -----------------------------
# Layer 8: Scoring Engine
# -----------------------------

def compute_consultation_score(extracted_symptoms: List[str], disease_matches: List[Dict], kb: List[Dict]) -> float:
	if not disease_matches:
		return 0.0

	top_disease_name = disease_matches[0]["disease"]
	top_disease = next((d for d in kb if d["disease"] == top_disease_name), None)
	if not top_disease:
		return 0.0

	expected = set(top_disease["symptoms"])
	matched = expected.intersection(set(extracted_symptoms))
	if not expected:
		return 0.0

	score = (len(matched) / len(expected)) * 100.0
	return round(score, 2)


# -----------------------------
# Layer 9: Output Formatting Layer
# -----------------------------

def build_final_output(
	transcription: Dict,
	symptoms_detected: List[str],
	session_symptoms: List[str],
	possible_conditions: List[Dict],
	missed_considerations: Dict,
	consultation_score: float,
	patient_transcript: Optional[List[str]] = None,
	doctor_transcript: Optional[List[str]] = None,
	doctor_actions: Optional[List[str]] = None,
	condition_checklist: Optional[List[Dict]] = None,
	unaddressed_conditions: Optional[List[str]] = None,
	patient_checklist: Optional[List[Dict]] = None,
	doctor_checklist: Optional[List[Dict]] = None,
	follow_up_questions: Optional[List[str]] = None,
	communication_bias: Optional[Dict] = None,
	dialog_turns: Optional[List[Dict[str, str]]] = None,
) -> Dict:
	return {
		"transcription": transcription,
		"symptoms_detected": symptoms_detected,
		"session_symptoms": session_symptoms,
		"possible_conditions": possible_conditions,
		"missed_considerations": missed_considerations,
		"consultation_score": consultation_score,
		"patient_transcript": patient_transcript or [],
		"doctor_transcript": doctor_transcript or [],
		"doctor_actions": doctor_actions or [],
		"condition_checklist": condition_checklist or [],
		"unaddressed_conditions": unaddressed_conditions or [],
		"patient_checklist": patient_checklist or [],
		"doctor_checklist": doctor_checklist or [],
		"follow_up_questions": follow_up_questions or [],
		"communication_bias": communication_bias or {},
		"dialog_turns": dialog_turns or [],
	}


# -----------------------------
# Pipeline Orchestration
# -----------------------------

def init_session_state() -> Dict:
	return {
		"all_patient_text": [],
		"all_doctor_text": [],
		"patient_turns": [],
		"doctor_turns": [],
		"all_symptoms": set(),
		"doctor_actions": set(),
		"processed_chunks": 0,
		"last_speaker": None,
		"last_result": None,
	}


def update_session_memory(session_state: Dict, speaker: str, text: str, symptoms: List[str]) -> None:
	session_state["last_speaker"] = speaker

	if not text:
		return

	if speaker == "PATIENT":
		session_state["patient_turns"].append(text)
		session_state["all_patient_text"].append(text)
	else:
		session_state["doctor_turns"].append(text)
		session_state["all_doctor_text"].append(text)
		session_state["doctor_actions"].update(extract_doctor_actions(text))

	for symptom in symptoms:
		session_state["all_symptoms"].add(symptom)


def process_transcription_result(
	transcription: Dict,
	kb: List[Dict],
	chunk_index: int,
	session_state: Dict,
	forced_speaker: Optional[str] = None,
) -> Optional[Dict]:
	full_text = str(transcription.get("full_text", "")).strip()
	if not full_text:
		return None

	avg_conf = average_confidence(transcription)
	if avg_conf < MIN_AVG_CONFIDENCE:
		return None

	if not forced_speaker and is_multi_party_text(full_text):
		return process_text_conversation(full_text, kb, session_state)

	speaker = forced_speaker or assign_speaker_label(chunk_index, full_text, session_state)
	structured_transcription = structure_transcript(transcription, speaker)

	# Process only patient speech for symptom extraction.
	patient_text = structured_transcription["full_text"] if speaker == "PATIENT" else ""
	symptoms = extract_symptoms(patient_text)

	update_session_memory(session_state, speaker, patient_text, symptoms)
	session_symptoms = sorted(session_state["all_symptoms"])

	# Analyze against cumulative patient evidence for better continuity.
	disease_matches = match_diseases(session_symptoms, kb)
	blindspots = detect_blindspots(session_symptoms, disease_matches)
	score = compute_consultation_score(session_symptoms, disease_matches, kb)

	doctor_actions = sorted(session_state.get("doctor_actions", set()))
	condition_checklist = build_condition_checklist(
		session_symptoms,
		disease_matches,
		kb,
		set(doctor_actions),
		session_state.get("all_doctor_text", []),
	)
	unaddressed_conditions = [
		item["disease"] for item in condition_checklist if item["status"] != "addressed"
	]

	communication_bias = detect_communication_bias(
		session_state.get("patient_turns", []),
		session_state.get("doctor_turns", []),
	)

	session_state["processed_chunks"] += 1

	final_output = build_final_output(
		transcription=structured_transcription,
		symptoms_detected=symptoms,
		session_symptoms=session_symptoms,
		possible_conditions=disease_matches,
		missed_considerations=blindspots,
		consultation_score=score,
		patient_transcript=session_state.get("patient_turns", []),
		doctor_transcript=session_state.get("doctor_turns", []),
		doctor_actions=doctor_actions,
		condition_checklist=condition_checklist,
		unaddressed_conditions=unaddressed_conditions,
		communication_bias=communication_bias,
	)

	session_state["last_result"] = final_output

	append_jsonl_log(
		{
			"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
			"chunk_index": chunk_index,
			"speaker": speaker,
			"avg_confidence": round(avg_conf, 3),
			"result": final_output,
		}
	)

	return final_output


def append_jsonl_log(record: Dict, log_path: str = SESSION_LOG_PATH) -> None:
	line = json.dumps(record, ensure_ascii=True)
	with open(log_path, "a", encoding="utf-8") as f:
		f.write(line + "\n")


def run_pipeline_once(model_state: Dict, kb: List[Dict], chunk_index: int, session_state: Dict) -> Optional[Dict]:
	audio = capture_speech_chunk()
	if audio is None or audio.size == 0:
		return None

	if DEBUG_SAVE_AUDIO:
		sf.write(DEBUG_AUDIO_PATH, audio, SAMPLE_RATE)

	try:
		transcription = transcribe_audio_chunk(model_state["model"], audio)
	except RuntimeError as e:
		error_text = str(e).lower()
		if model_state.get("device") == "cuda" and ("cublas" in error_text or "cuda" in error_text):
			print("[Whisper] CUDA runtime unavailable during inference. Falling back to CPU...")
			model_state["model"] = WhisperModel("base", device="cpu", compute_type="int8")
			model_state["device"] = "cpu"
			transcription = transcribe_audio_chunk(model_state["model"], audio)
		else:
			raise

	return process_transcription_result(transcription, kb, chunk_index, session_state)


def _resample_if_needed(audio: np.ndarray, source_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
	if source_sr == target_sr:
		return audio
	if audio.size == 0:
		return audio

	duration = len(audio) / float(source_sr)
	target_length = int(duration * target_sr)
	if target_length <= 0:
		return np.array([], dtype=np.float32)

	x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
	x_new = np.linspace(0.0, duration, num=target_length, endpoint=False)
	return np.interp(x_new, x_old, audio).astype(np.float32)


def create_app(model_state: Dict, kb: List[Dict], session_state: Dict) -> Flask:
	app = Flask(__name__)
	state_lock = threading.Lock()

	@app.after_request
	def add_cors_headers(response):
		response.headers["Access-Control-Allow-Origin"] = "*"
		response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
		response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
		return response

	@app.route("/health", methods=["GET"])
	def health():
		return jsonify({"status": "ok", "device": model_state.get("device")})

	@app.route("/api/session/reset", methods=["POST"])
	def reset_session():
		with state_lock:
			session_state.clear()
			session_state.update(init_session_state())
		return jsonify({"message": "session reset"})

	@app.route("/api/session/state", methods=["GET"])
	def get_session_state_api():
		with state_lock:
			payload = {
				"processed_chunks": session_state.get("processed_chunks", 0),
				"last_speaker": session_state.get("last_speaker"),
				"session_symptoms": sorted(session_state.get("all_symptoms", set())),
				"patient_transcript": session_state.get("patient_turns", []),
				"doctor_transcript": session_state.get("doctor_turns", []),
				"doctor_actions": sorted(session_state.get("doctor_actions", set())),
				"last_result": session_state.get("last_result"),
			}
		return jsonify(payload)

	@app.route("/api/analyze/text", methods=["POST"])
	def analyze_text():
		payload = request.get_json(silent=True) or {}
		text = str(payload.get("text", "")).strip()
		if not text:
			return jsonify({"error": "'text' is required"}), 400

		forced_speaker = str(payload.get("speaker", "")).strip().upper()
		if forced_speaker not in {"", "PATIENT", "DOCTOR"}:
			return jsonify({"error": "'speaker' must be PATIENT or DOCTOR when provided"}), 400

		if not forced_speaker:
			with state_lock:
				result = process_text_conversation(text, kb, session_state)
			if result is None:
				return jsonify({"message": "no result (empty input or processing issue)"}), 200
			return jsonify(result)

		transcription = {
			"segments": [
				{"id": 0, "start": 0.0, "end": 0.0, "text": text, "confidence": 1.0}
			],
			"full_text": text,
		}

		with state_lock:
			chunk_index = session_state.get("processed_chunks", 0)
			result = process_transcription_result(
				transcription,
				kb,
				chunk_index,
				session_state,
				forced_speaker=forced_speaker if forced_speaker else None,
			)

		if result is None:
			return jsonify({"message": "no result (below confidence threshold or empty input)"}), 200
		return jsonify(result)

	@app.route("/api/analyze/audio", methods=["POST"])
	def analyze_audio():
		if "audio" not in request.files:
			return jsonify({"error": "multipart field 'audio' is required"}), 400

		upload = request.files["audio"]
		binary = upload.read()
		if not binary:
			return jsonify({"error": "empty audio file"}), 400

		try:
			audio, source_sr = sf.read(io.BytesIO(binary), dtype="float32")
		except Exception as e:
			return jsonify({"error": f"unable to decode audio: {e}"}), 400

		if audio.ndim > 1:
			audio = np.mean(audio, axis=1)

		audio = _resample_if_needed(audio.astype(np.float32), source_sr, SAMPLE_RATE)
		if audio.size == 0:
			return jsonify({"error": "decoded audio is empty"}), 400

		try:
			transcription = transcribe_audio_chunk(model_state["model"], audio)
		except RuntimeError as e:
			error_text = str(e).lower()
			if model_state.get("device") == "cuda" and ("cublas" in error_text or "cuda" in error_text):
				model_state["model"] = WhisperModel("base", device="cpu", compute_type="int8")
				model_state["device"] = "cpu"
				transcription = transcribe_audio_chunk(model_state["model"], audio)
			else:
				return jsonify({"error": str(e)}), 500

		with state_lock:
			chunk_index = session_state.get("processed_chunks", 0)
			result = process_transcription_result(transcription, kb, chunk_index, session_state)

		if result is None:
			return jsonify({"message": "no result (below confidence threshold or empty transcription)"}), 200
		return jsonify(result)

	@app.route("/api/analyze/live", methods=["POST"])
	def analyze_live_from_backend_mic():
		with state_lock:
			chunk_index = session_state.get("processed_chunks", 0)
			result = run_pipeline_once(model_state, kb, chunk_index, session_state)

		if result is None:
			return jsonify({"message": "no result (no speech detected or below confidence threshold)"}), 200
		return jsonify(result)

	return app


def run_cli_loop(model_state: Dict, kb: List[Dict], session_state: Dict) -> None:
	print("System ready. Speak into the microphone. Press Ctrl+C to exit.")
	print(f"Whisper device selected: {model_state['device']}")
	print(f"Speaker mode: {SPEAKER_MODE}")
	print(f"Confidence threshold: {MIN_AVG_CONFIDENCE}")
	print(f"JSONL logging path: {SESSION_LOG_PATH}")
	chunk_index = 0

	try:
		while True:
			result = run_pipeline_once(model_state, kb, chunk_index, session_state)
			chunk_index += 1

			if result is None:
				continue

			print("\n===== Clinical Insight =====")
			print(json.dumps(result, indent=2))

	except KeyboardInterrupt:
		print("\nShutting down Clinical Blindspot AI.")


def run_api_server(model_state: Dict, kb: List[Dict], session_state: Dict, host: str, port: int) -> None:
	app = create_app(model_state, kb, session_state)
	print("Clinical Blindspot API server started")
	print(f"Whisper device selected: {model_state['device']}")
	print(f"Listening on http://{host}:{port}")
	print("Endpoints: /health, /api/analyze/text, /api/analyze/audio, /api/analyze/live, /api/session/state, /api/session/reset")
	app.run(host=host, port=port, debug=False, threaded=True)


def main() -> None:
	parser = argparse.ArgumentParser(description="Clinical Blindspot AI")
	parser.add_argument("--mode", choices=["cli", "server"], default="cli", help="Run as terminal loop or HTTP API server")
	parser.add_argument("--host", default="127.0.0.1", help="API host when mode=server")
	parser.add_argument("--port", type=int, default=8000, help="API port when mode=server")
	args = parser.parse_args()

	print("Clinical Blindspot AI: initializing model and medical knowledge base...")
	if WEBRTCVAD_AVAILABLE:
		print("VAD engine: webrtcvad")
	else:
		print("VAD engine: energy fallback (webrtcvad unavailable)")
	print(f"Whisper model source: {'local folder' if os.path.isdir(LOCAL_WHISPER_MODEL_DIR) else 'online/base cache'}")
	try:
		model, model_device = init_whisper_model()
	except RuntimeError as e:
		print(f"[Startup error] {e}")
		return
	model_state = {"model": model, "device": model_device}
	kb = load_medical_knowledge_base()
	session_state = init_session_state()

	if args.mode == "server":
		run_api_server(model_state, kb, session_state, args.host, args.port)
	else:
		run_cli_loop(model_state, kb, session_state)


if __name__ == "__main__":
	main()

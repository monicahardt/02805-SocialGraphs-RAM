"""Heuristically extract and attribute dialogue from the HP1 plain-text book."""

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "book_data" / "hp1_book.txt"
DEFAULT_OUTPUT = ROOT / "book_data" / "hp1_dialogue_v2.csv"
BOOK_TITLE = "Harry Potter and the Philosopher's Stone"

# Surface forms in the novel mapped to one canonical name. Extend this table when
# the review file shows a recurring form that is not recognized.
ALIASES = {
    "Albus Dumbledore": ["Albus Dumbledore", "Professor Dumbledore", "Dumbledore"],
    "Minerva McGonagall": ["Minerva McGonagall", "Professor McGonagall", "McGonagall"],
    "Rubeus Hagrid": ["Rubeus Hagrid", "Professor Hagrid", "Hagrid"],
    "Harry Potter": ["Harry Potter", "Harry"],
    "Hermione Granger": ["Hermione Granger", "Hermione"],
    "Ron Weasley": ["Ron Weasley", "Ron"],
    "Draco Malfoy": ["Draco Malfoy", "Malfoy"],
    "Neville Longbottom": ["Neville Longbottom", "Neville"],
    "Severus Snape": ["Severus Snape", "Professor Snape", "Snape"],
    "Quirinus Quirrell": ["Quirinus Quirrell", "Professor Quirrell", "Quirrell"],
    "Voldemort": ["Lord Voldemort", "Voldemort", "You-Know-Who"],
    "Petunia Dursley": ["Mrs. Dursley", "Mrs Dursley", "Petunia Dursley", "Petunia"],
    "Vernon Dursley": ["Mr. Dursley", "Mr Dursley", "Vernon Dursley", "Vernon"],
    "Dudley Dursley": ["Dudley Dursley", "Dudley"],
    "Oliver Wood": ["Oliver Wood", "Wood"],
    "Percy Weasley": ["Percy Weasley", "Percy"],
    "Fred Weasley": ["Fred Weasley", "Fred"],
    "George Weasley": ["George Weasley", "George"],
    "Molly Weasley": ["Molly Weasley", "Mrs. Weasley", "Mrs Weasley"],
    "Lee Jordan": ["Lee Jordan", "Lee"],
    "Argus Filch": ["Argus Filch", "Filch"],
    "Nearly Headless Nick": ["Nearly Headless Nick", "Nick"],
    "Sorting Hat": ["Sorting Hat"],
    "Madam Hooch": ["Madam Hooch", "Madam Hooch"],
    "Filius Flitwick": ["Professor Flitwick", "Flitwick"],
    "Pomona Sprout": ["Professor Sprout", "Sprout"],
    "Madam Pomfrey": ["Madam Pomfrey", "Madam Pomfrey"],
    "Ollivander": ["Ollivander"],
    "Griphook": ["Griphook"],
    "Firenze": ["Firenze"],
    "Bane": ["Bane"],
    "Fang": ["Fang"],
    # The broadcast speakers in chapter one are identified by role, not name.
    "Newsreader": ["newscaster"],
    "Weatherman": ["weatherman"],
}

SPEECH_VERBS = (
    "said|asked|replied|whispered|shouted|yelled|cried|called|muttered|mumbled|answered|"
    "exclaimed|snapped|barked|groaned|moaned|laughed|gasped|continued|added|"
    "interrupted|demanded|suggested|stammered|sobbed|hissed|screamed|began|"
    "remarked|insisted|inquired|told|argued|warned|pleaded|begged|announced|"
    # Common dialogue tags and inflected forms that occur in the OCR book files.
    "chortled|grunted|snorted|scoffed|retorted|growled|roared|bellowed|boomed|"
    "murmured|whined|whimpered|wailed|shrieked|squealed|yelped|jeered|urged|"
    "repeated|protested|explained|reassured|complained|warned|ranted|drawled|"
    "yelling|bellowing|roaring|chortling|grunting|murmuring|"
    "speaking|instructed|ordered|commanded|stated|declared|announced"
)
VERB_RE = re.compile(rf"\b(?:{SPEECH_VERBS})\b", re.I)
QUOTE_RE = re.compile(r"“([^”]*)”|\"([^\"]*)\"")

ALIAS_TO_NAME = {}
for canonical, aliases in ALIASES.items():
    for alias in aliases:
        ALIAS_TO_NAME[alias.lower()] = canonical
ALIAS_PATTERN = "|".join(
    re.escape(alias) for alias in sorted(ALIAS_TO_NAME, key=len, reverse=True)
)
ALIAS_RE = re.compile(rf"(?<!\w)(?P<alias>{ALIAS_PATTERN})(?!\w)", re.I)


def chapter_sections(text):
    headings = list(re.finditer(r"(?m)^\s*CHAPTER\s+(?:\d+|[A-Z]+)\b[^\n]*$", text))
    if not headings:
        # Some of the OCR text files lost the word "CHAPTER" and its number.
        # In those files, chapter titles are standalone all-caps lines after a
        # page footer (occasionally preceded by the chapter number).
        lines = text.splitlines(keepends=True)
        offsets = []
        offset = 0
        for line in lines:
            offsets.append(offset)
            offset += len(line)

        candidates = []
        page_footer = re.compile(r"^\s*Page\s*\|\s*\d+\b", re.I)
        title_line = re.compile(r"^[A-Z][A-Z ’'&-]*(?:\s+[A-Z][A-Z ’'&-]*)*\s*$")
        for i, line in enumerate(lines):
            if not title_line.match(line.strip()) or len(re.findall(r"[A-Z]+", line)) < 1:
                continue
            # Combine a title broken over consecutive OCR lines.
            end = i + 1
            while end < len(lines) and title_line.match(lines[end].strip()):
                end += 1
            prior_lines = [part.strip() for part in lines[max(0, i - 12):i] if part.strip()]
            has_page_footer = any(page_footer.match(part) for part in prior_lines)
            has_chapter_number = any(part.isdigit() for part in prior_lines[-3:])
            starts_book = not candidates and offsets[i] < 1200
            if not (has_page_footer or has_chapter_number or starts_book):
                continue
            after = end
            while after < len(lines) and not lines[after].strip():
                after += 1
            if after >= len(lines) or title_line.match(lines[after].strip()):
                continue
            # Skip the footer and optional printed chapter number in the section.
            start = offsets[i]
            candidates.append((start, " ".join(part.strip() for part in lines[i:end])))

        if not candidates:
            raise ValueError("No chapter headings found (expected CHAPTER lines or standalone all-caps titles).")
        sections = []
        for index, (start, title) in enumerate(candidates):
            end = candidates[index + 1][0] if index + 1 < len(candidates) else len(text)
            # The title starts at this offset; skip its line(s) and following blanks.
            line_index = next(j for j, line_offset in enumerate(offsets) if line_offset == start)
            next_index = line_index
            while next_index < len(lines) and title_line.match(lines[next_index].strip()):
                next_index += 1
            while next_index < len(lines) and not lines[next_index].strip():
                next_index += 1
            body_start = offsets[next_index] if next_index < len(offsets) else len(text)
            sections.append((title, text[body_start:end]))
        return sections
    sections = []
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        lines = text[match.end():end].splitlines()
        nonempty = next((i for i, line in enumerate(lines) if line.strip()), None)
        title = lines[nonempty].strip() if nonempty is not None else match.group().strip()
        body = "\n".join(lines[nonempty + 1:]) if nonempty is not None else ""
        sections.append((title, body))
    return sections


def canonical_alias(match):
    return ALIAS_TO_NAME[match.group("alias").lower()]


def most_recent_named_character(text):
    matches = list(ALIAS_RE.finditer(text))
    return canonical_alias(matches[-1]) if matches else None


def gender_for(name):
    if name in {"Petunia Dursley", "Minerva McGonagall", "Hermione Granger", "Molly Weasley", "Madam Hooch", "Pomona Sprout", "Madam Pomfrey"}:
        return "she"
    if name in {"Vernon Dursley", "Albus Dumbledore", "Rubeus Hagrid", "Harry Potter", "Ron Weasley", "Draco Malfoy", "Neville Longbottom", "Severus Snape", "Quirinus Quirrell", "Dudley Dursley", "Oliver Wood", "Percy Weasley", "Fred Weasley", "George Weasley", "Lee Jordan", "Argus Filch", "Filius Flitwick", "Ollivander", "Griphook", "Firenze", "Bane"}:
        return "he"
    return None


def explicit_speaker(around_quote, quote_start, quote_end):
    """Find a named speech tag close to the quote, preferring the closest one."""
    candidates = []
    prefix = around_quote[max(0, quote_start - 130):quote_start]
    suffix = around_quote[quote_end:min(len(around_quote), quote_end + 130)]
    # A tag after the previous quote can identify a continuation in the same
    # paragraph ("Well, Ted," said the weatherman, "I don't know...").
    previous_quotes = list(QUOTE_RE.finditer(around_quote[:quote_start]))
    if previous_quotes:
        prefix = around_quote[previous_quotes[-1].end():quote_start]
    patterns = [
        (re.compile(rf"(?P<alias>{ALIAS_PATTERN})\b[^.!?]{{0,45}}?\b(?:{SPEECH_VERBS})\b", re.I), "before"),
        (re.compile(rf"\b(?:{SPEECH_VERBS})\b[^.!?]{{0,45}}?(?P<alias>{ALIAS_PATTERN})\b", re.I), "before"),
        (re.compile(rf"\b(?:{SPEECH_VERBS})\b[^.!?]{{0,45}}?(?P<alias>{ALIAS_PATTERN})\b", re.I), "after"),
        (re.compile(rf"(?P<alias>{ALIAS_PATTERN})\b[^.!?]{{0,45}}?\b(?:{SPEECH_VERBS})\b", re.I), "after"),
    ]
    for pattern, side in patterns:
        chunk = prefix if side == "before" else suffix
        for match in pattern.finditer(chunk):
            candidates.append((abs(len(prefix) - match.end()) if side == "before" else match.start(),
                               ALIAS_TO_NAME[match.group("alias").lower()]))
    return min(candidates)[1] if candidates else None


def pronoun_speaker(around_quote, quote_start, quote_end, prior_names=()):
    prefix = around_quote[max(0, quote_start - 100):quote_start]
    suffix = around_quote[quote_end:min(len(around_quote), quote_end + 100)]
    pronoun_tag = re.compile(rf"\b(he|she)\b[^.!?]{{0,35}}?\b(?:{SPEECH_VERBS})\b", re.I)
    suffix_tag = pronoun_tag.search(suffix)
    prefix_tag = pronoun_tag.search(prefix)
    if suffix_tag:
        pronoun = suffix_tag.group(1).lower()
        names = list(ALIAS_RE.finditer(prefix))
        for match in reversed(names):
            name = canonical_alias(match)
            if gender_for(name) == pronoun:
                return name
        # A sentence may identify the pronoun's subject after the speech tag:
        # “Sorry,” he grunted ... before Mr. Dursley realized ...
        names = list(ALIAS_RE.finditer(suffix[suffix_tag.end():]))
        for match in names:
            name = canonical_alias(match)
            if gender_for(name) == pronoun:
                return name
    if prefix_tag:
        pronoun = prefix_tag.group(1).lower()
        names = list(ALIAS_RE.finditer(prefix[:prefix_tag.start()]))
        for match in reversed(names):
            name = canonical_alias(match)
            if gender_for(name) == pronoun:
                return name
    # A nearby action beat such as "He cleared his throat" can identify the
    # speaker even when the narration omits "he said".
    action = re.search(r"\b(he|she)\b[^.!?]{0,55}\b(?:cleared|raised|turned|looked|smiled|nodded|sighed|laughed|called)\b", prefix, re.I)
    if action:
        pronoun = action.group(1).lower()
        names = list(ALIAS_RE.finditer(prefix[:action.start()]))
        for match in reversed(names):
            name = canonical_alias(match)
            if gender_for(name) == pronoun:
                return name
    return None


def contextual_speaker(paragraph, quote_start, quote_end, speech, prior_names=()):
    """Use direct local narration cues that do not follow a speech-tag form."""
    prefix = paragraph[max(0, quote_start - 360):quote_start]
    # The narration explicitly says Dudley learned the quoted word.
    learned_word = re.search(
        rf"(?P<alias>{ALIAS_PATTERN})\b[^.!?]{{0,35}}?learned a new word\s*\($",
        prefix,
        re.I,
    )
    if learned_word:
        return ALIAS_TO_NAME[learned_word.group("alias").lower()]

    # In the evening-news passage, the narration identifies the speaker by role.
    if re.search(r"(?:report on the )?evening news\s*:\s*$", prefix, re.I):
        return "Newsreader"
    if re.search(r"\bnewscaster\b[^.!?]{0,80}[.!?]?\s*$", prefix, re.I):
        return "Newsreader"
    if re.search(r"\bweatherman\b[^.!?]{0,80}$", prefix, re.I):
        return "Weatherman"

    if re.search(r"\bMrs\.?\s+Dursley looked shocked and angry\b", prefix, re.I):
        return "Petunia Dursley"

    if re.search(r"tiny old man[\s\S]{0,260}\bhe said in a squeaky voice\b", prefix, re.I):
        return "Albus Dumbledore"

    # Resolve an immediately preceding action beat from the named character
    # context in the chapter, while avoiding stale resolution from “he said”.
    action = re.search(
        r"\b(he|she)\b[^.!?]{0,55}\b(?:cleared|raised|turned|looked|smiled|nodded|sighed|laughed)\b[^.!?]*[.!?]?\s*$",
        prefix,
        re.I,
    )
    if action:
        pronoun = action.group(1).lower()
        local_names = list(ALIAS_RE.finditer(prefix))
        if local_names:
            recent_name = canonical_alias(local_names[-1])
            if pronoun == "he" and recent_name == "Petunia Dursley":
                return "Vernon Dursley"
            if pronoun == "she" and recent_name == "Vernon Dursley":
                return "Petunia Dursley"
        for name in reversed(prior_names):
            if gender_for(name) == pronoun:
                return name
    return None


def self_identified_speaker(speech):
    """Recognize short self-identifications such as “Vernon Dursley speaking.”"""
    match = re.match(
        rf"\s*(?P<alias>{ALIAS_PATTERN})\s+(?:speaking|here|this is)\b",
        speech,
        re.I,
    )
    return ALIAS_TO_NAME[match.group("alias").lower()] if match else None


def extract_chapter(chapter, body):
    rows = []
    previous_speakers = []
    previous_quote_end = None
    previous_speaker = None
    prior_names = []
    recent_narration = ""
    source_offset = 0
    for raw_paragraph in re.split(r"(\n\s*\n)", body):
        if not raw_paragraph.strip():
            source_offset += len(raw_paragraph)
            continue
        paragraph = re.sub(r"\s+", " ", raw_paragraph).strip()
        masked_narration = QUOTE_RE.sub(lambda m: " " * len(m.group(0)), paragraph)
        for quote in QUOTE_RE.finditer(paragraph):
            speech = (quote.group(1) or quote.group(2) or "").strip()
            if not speech:
                continue
            # Do not carry a turn-taking guess across a long stretch of narration.
            global_quote_start = source_offset + quote.start()
            for alias_match in ALIAS_RE.finditer(masked_narration[:quote.start()]):
                name = canonical_alias(alias_match)
                if not prior_names or prior_names[-1] != name:
                    prior_names.append(name)
            prior_names = prior_names[-4:]
            if previous_quote_end is not None and global_quote_start - previous_quote_end > 650:
                previous_speakers = []
            speaker = explicit_speaker(paragraph, quote.start(), quote.end())
            basis = "named speech tag" if speaker else ""
            confidence = "high" if speaker else ""
            if not speaker:
                context = f"{recent_narration} {paragraph[:quote.start()]}"
                context_end = len(recent_narration) + 1 + quote.start()
                speaker = contextual_speaker(context, context_end, context_end, speech, prior_names)
                if speaker:
                    basis, confidence = "speaker identified by nearby narration", "high"
            if not speaker:
                speaker = self_identified_speaker(speech)
                if speaker:
                    basis, confidence = "speaker identified in dialogue", "high"
            if not speaker:
                speaker = pronoun_speaker(paragraph, quote.start(), quote.end(), prior_names)
                if speaker:
                    basis, confidence = "pronoun resolved to nearby named character", "medium"
            between_turns = paragraph[max(0, quote.start() - 75):quote.start()]
            has_pronoun_tag = re.search(
                rf"\b(?:he|she)\b[^.!?]{{0,35}}?\b(?:{SPEECH_VERBS})\b", between_turns, re.I
            )
            if not speaker and previous_speaker and previous_quote_end is not None \
                    and global_quote_start - previous_quote_end < 70 and has_pronoun_tag:
                speaker = previous_speaker
                basis, confidence = "same speaker continuing a tagged turn", "low"
            if not speaker and previous_quote_end is not None \
                    and global_quote_start - previous_quote_end < 650 \
                    and len(previous_speakers) >= 2 and previous_speakers[-1] != previous_speakers[-2]:
                speaker = previous_speakers[-2]
                basis, confidence = "alternating turns between recent speakers", "low"
            if speaker:
                if not previous_speakers or previous_speakers[-1] != speaker:
                    previous_speakers.append(speaker)
                    previous_speakers = previous_speakers[-2:]
                previous_speaker = speaker
            else:
                speaker, basis, confidence = "Unknown", "no reliable attribution", "unattributed"
            rows.append({
                "movie": BOOK_TITLE,
                "chapter": chapter,
                "character": speaker,
                "dialog": re.sub(r"\s+", " ", speech),
                "attribution": basis,
                "confidence": confidence,
            })
            previous_quote_end = global_quote_start + quote.end() - quote.start()
        for alias_match in ALIAS_RE.finditer(masked_narration):
            name = canonical_alias(alias_match)
            if not prior_names or prior_names[-1] != name:
                prior_names.append(name)
        prior_names = prior_names[-4:]
        recent_narration = f"{recent_narration} {masked_narration}"[-360:]
        source_offset += len(raw_paragraph)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8-sig")
    rows = [row for chapter, body in chapter_sections(text) for row in extract_chapter(chapter, body)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["movie", "chapter", "character", "dialog"],
            extrasaction="ignore",
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    attributed = sum(row["character"] != "Unknown" for row in rows)
    print(f"Wrote {total:,} dialogue candidates to {args.output}")
    print(f"Attributed: {attributed:,} ({attributed / total:.1%})" if total else "No dialogue candidates found.")
    print("Review rows with confidence 'low' and 'unattributed' before making a book network.")


if __name__ == "__main__":
    main()

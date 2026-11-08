"""Generate a small, ORIGINAL demo show in the same screenplay format as the real input.

The real scripts are confidential, so the public demo runs on this made-up show instead.
Run: python demo/make_demo_show.py   (writes demo/the_contract_bride/ep01.txt ... ep03.txt)
"""

from pathlib import Path

OUT = Path(__file__).parent / "the_contract_bride"

# Each scene: (heading, [beats]). A beat: (kind, speaker, emotions, text, seconds).
# kind: C=camera, A=action, S=sound fx, D=dialogue. Camera beats run alongside the next beat.
EPISODES = [
    {
        "logline": "At her stepsister's engagement party, waitress Maya is humiliated and fired - until the "
                   "groom's rival, billionaire Daniel Reyes, walks in and claims her as his wife.",
        "continuity": "First episode. Establishes Maya, her stepfamily, and the secret contract marriage.",
        "scenes": [
            ("INT. GRAND HOTEL BALLROOM - NIGHT", [
                ("C", "", "", "Wide shot of a glittering ballroom; Maya weaves between guests with a silver tray.", 3),
                ("A", "", "", "Vanessa steps into Maya's path and knocks the tray from her hands. Glasses shatter.", 3),
                ("S", "", "", "Crash of breaking glass; the music stops.", 2),
                ("D", "VANESSA", "Mocking, Loud", "Look, everyone - my sister, the help.", 3),
                ("D", "MAYA", "Shaken", "Vanessa, please. Not here.", 3),
                ("D", "VANESSA", "Cruel", "Not here? You don't belong anywhere near here.", 4),
                ("C", "", "", "Close-up on Maya's trembling hands gripping her apron.", 2),
                ("D", "LINDA", "Cold, Dismissive", "Security will see you out. You're fired, Maya.", 4),
                ("D", "MAYA", "Desperate", "You can't fire me. I need this job for Dad's treatment.", 4),
                ("D", "VANESSA", "Gloating", "Then maybe beg.", 2),
                ("A", "", "", "Vanessa slaps Maya hard across the face. Guests gasp.", 3),
                ("S", "", "", "A sharp slap echoes through the silent ballroom.", 1),
                ("D", "MAYA", "Stunned, Furious", "You will regret that.", 3),
                ("D", "VANESSA", "Laughing", "Who's going to make me? You have nobody.", 4),
            ]),
            ("INT. GRAND HOTEL BALLROOM - CONTINUOUS", [
                ("C", "", "", "The ballroom doors swing open; a tall man in a black suit stands framed in light.", 3),
                ("A", "", "", "Daniel Reyes walks straight through the crowd, which parts in stunned silence.", 4),
                ("D", "LINDA", "Shocked", "Mr. Reyes? What are you doing here?", 3),
                ("D", "DANIEL", "Calm, Dangerous", "Who touched her?", 3),
                ("C", "", "", "Close-up on Vanessa's smile freezing.", 2),
                ("D", "VANESSA", "Nervous", "She's just a waitress. She's nobody.", 3),
                ("A", "", "", "Daniel takes off his jacket and places it around Maya's shoulders.", 3),
                ("D", "DANIEL", "Cold, Final", "She's not nobody. She's my wife.", 4),
                ("A", "", "", "Linda drops her champagne glass. Vanessa staggers back, eyes wide in shock.", 3),
                ("C", "", "", "Cut to black.", 1),
            ]),
        ],
    },
    {
        "logline": "Linda threatens to expose the contract marriage, and Maya discovers her late mother's will "
                   "was forged to cut her out.",
        "continuity": "Picks up right after Daniel's announcement at the party.",
        "scenes": [
            ("INT. HOTEL CORRIDOR - NIGHT", [
                ("A", "", "", "Linda grabs Maya's wrist and drags her into an empty corridor.", 3),
                ("D", "LINDA", "Furious, Low", "A contract marriage? Do you think I'm stupid?", 4),
                ("D", "MAYA", "Defiant", "How do you know about the contract?", 3),
                ("D", "LINDA", "Threatening", "Leave him, or the whole city will know he bought you.", 5),
                ("D", "MAYA", "Steady", "Then tell them. I'm done being afraid of you.", 4),
                ("D", "LINDA", "Sinister", "Brave words. Your father's hospital bills say otherwise.", 4),
                ("A", "", "", "Linda shoves an envelope into Maya's hands and walks away.", 3),
            ]),
            ("INT. DANIEL'S PENTHOUSE - LATER", [
                ("C", "", "", "Maya sits alone at a long glass table, the envelope open in front of her.", 3),
                ("A", "", "", "Maya unfolds a document: her mother's will. Her hands start to shake.", 3),
                ("D", "MAYA", "Confused", "This isn't Mom's handwriting.", 3),
                ("D", "DANIEL", "Concerned", "What is it?", 2),
                ("D", "MAYA", "Trembling", "Mom left the house to me. This says she left everything to Linda.", 5),
                ("D", "DANIEL", "Grave", "Then someone forged it.", 3),
                ("A", "", "", "Daniel takes the will and holds it to the light. A faint second signature shows through.", 4),
                ("D", "DANIEL", "Sharp", "There's another name under this one.", 3),
                ("D", "MAYA", "Breathless", "Who signed this?", 3),
                ("C", "", "", "Extreme close-up on the hidden signature, too faint to read.", 2),
            ]),
        ],
    },
    {
        "logline": "Vanessa frames Maya for stealing a diamond necklace at a charity gala, but the security "
                   "footage reveals a betrayal no one saw coming.",
        "continuity": "One week later. Maya is now publicly known as Daniel's wife.",
        "scenes": [
            ("INT. CHARITY GALA - NIGHT", [
                ("A", "", "", "Vanessa screams and points at Maya across the crowded gala.", 3),
                ("D", "VANESSA", "Hysterical", "My necklace! She stole my diamond necklace!", 4),
                ("D", "MAYA", "Shocked", "What? I haven't been anywhere near you.", 3),
                ("A", "", "", "A security guard grabs Maya's purse and turns it upside down.", 3),
                ("S", "", "", "A diamond necklace clatters onto the marble floor.", 2),
                ("D", "GUARD", "Stern", "Ma'am, you'll need to come with us.", 3),
                ("D", "VANESSA", "Triumphant", "Once a thief, always a thief.", 3),
                ("D", "MAYA", "Furious", "I didn't take it, and you know that. Check the cameras.", 4),
            ]),
            ("INT. SECURITY OFFICE - CONTINUOUS", [
                ("C", "", "", "Rows of monitors glow in a dark security office.", 2),
                ("A", "", "", "Daniel leans over the guard's shoulder as the footage rewinds.", 3),
                ("D", "DANIEL", "Cold", "Stop. Go back ten seconds.", 3),
                ("C", "", "", "On screen: a figure slips the necklace into Maya's purse.", 3),
                ("D", "GUARD", "Surprised", "That's not the sister.", 2),
                ("D", "MAYA", "Disbelieving", "No. That's impossible.", 3),
                ("A", "", "", "The figure on screen turns toward the camera. Maya gasps and steps back in horror.", 4),
                ("D", "MAYA", "Devastated", "Dad?", 2),
                ("D", "DANIEL", "Grave", "Your father planted it. The question is why.", 4),
            ]),
        ],
    },
]


def ts(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def render(n: int, ep: dict) -> str:
    out = [f"EPISODE NUMBER: Episode {n}", f"EPISODE LOGLINE: {ep['logline']}",
           f"CONTINUITY NOTES: {ep['continuity']}", "", "--- SCRIPT START ---", ""]
    t = 0.0
    for i, (heading, beats) in enumerate(ep["scenes"], 1):
        start = t
        blocks = []
        for kind, speaker, emo, text, dur in beats:
            if kind == "C":       # camera shot overlapping the next beat
                blocks.append(("**CAMERA:**", t, t + dur, text))
                t += dur * 0.5
                continue
            label = {"A": "**ACTION:**", "S": "**SOUND FX:**"}.get(kind) or f"**{speaker} ({emo}):**"
            body = f'"{text}"' if kind == "D" else text
            blocks.append((label, t + 0.2, t + dur, body))
            t += dur + 0.3
        t = round(t, 2)
        out += [f"SCENE {i:02d}", f"Scene heading: {heading.replace(' - ', ' – ')}",
                f"Scene start: {ts(start)}", f"Scene end: {ts(t)}", "", "Actions and dialogue:", ""]
        for label, a, b, body in blocks:
            out += [label, f"Start: ({ts(a)}) | End: ({ts(min(b, t))})", body, ""]
    out.append("--- SCRIPT END ---")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for n, ep in enumerate(EPISODES, 1):
        (OUT / f"ep{n:02d}.txt").write_text(render(n, ep), encoding="utf-8")
    print(f"Wrote {len(EPISODES)} episodes to {OUT}")

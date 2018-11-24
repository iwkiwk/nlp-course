import random


def variable_p(pattern):
    return type(pattern) is str and len(pattern) > 1 and pattern.startswith('?')


def segment_p(pattern):
    return type(pattern) is list and pattern and len(pattern[0]) > 2 and pattern[0][0:2] == '?*'


def consp(pattern):
    return type(pattern) is list and len(pattern) > 0


def pat_match(pattern, inputs, bindings=None):
    if bindings is False:
        return False
    if pattern == inputs:
        return bindings
    bindings = bindings or {}

    if segment_p(pattern):
        token = pattern[0]
        var = token[2:]
        return segment_match(var, pattern[1:], inputs, bindings)
    elif variable_p(pattern):
        return match_variable(pattern[1:], [inputs], bindings)
    elif consp(pattern) and consp(inputs):
        return pat_match(pattern[1:], inputs[1:], pat_match(pattern[0], inputs[0], bindings))
    else:
        return False


def match_variable(var, inputs, bindings):
    binding = bindings.get(var)
    if not binding:
        bindings.update({var: inputs})
        return bindings
    if inputs == bindings[var]:
        return bindings
    return False


def segment_match(var, pattern, inputs, bindings, start=0):
    if not pattern:
        return match_variable(var, inputs, bindings)
    word = pattern[0]
    try:
        pos = start + inputs[start:].index(word)
    except ValueError:
        return False

    var_match = match_variable(var, inputs[:pos], dict(bindings))
    match = pat_match(pattern, inputs[pos:], var_match)

    if not match:
        return segment_match(var, pattern, inputs, bindings, start + 1)

    return match


def eliza(rules, defaults):
    while True:
        inputs = input('I > ').upper()
        if inputs == '[ABORT]':
            break
        if not inputs:
            continue
        print('ELIZA > ', use_eliza_rules(rules, inputs, defaults))


def use_eliza_rules(rules, inputs, defaults):
    inputs = inputs.split()

    matching_rules = []
    for pattern, matches in rules:
        pattern = pattern.split()
        replaces = pat_match(pattern, inputs)
        if replaces:
            matching_rules.append((matches, replaces))

    if matching_rules:
        responses, replaces = random.choice(matching_rules)
        response = random.choice(responses)
    else:
        replaces = {}
        response = random.choice(defaults)

    for var, replacement in replaces.items():
        replacement = ' '.join(switch_viewpoint(replacement))
        if replacement:
            response = response.replace('?' + var, replacement)

    return response


def replace(word, replacements):
    for old, new in replacements:
        if word == old:
            return new
    return word


def switch_viewpoint(words):
    switches = [('I', 'YOU'),
                ('YOU', 'I'),
                ('ME', 'YOU'),
                ('AM', 'ARE')]
    return [replace(w, switches) for w in words]


eliza_rules = {
    "?*x hello ?*y": [
        "How do you do. Please state your problem."
    ],
    "?*x I want ?*y": [
        "What would it mean if you got ?y?",
        "Why do you want ?y?",
        "Suppose you got ?y soon."
    ],
    "?*x if ?*y": [
        "Do you really think it's likely that ?y?",
        "Do you wish that ?y?",
        "What do you think about ?y?",
        "Really-- if ?y?"
    ],
    "?*x no ?*y": [
        "Why not?",
        "You are being a bit negative.",
        'Are you saying "No" just to be negative?'
    ],
    "?*x I was ?*y": [
        "Were you really?",
        "Perhaps I already knew you were ?y.",
        "Why do you tell me you were ?y now?"
    ],
    "?*x I feel ?*y": [
        "Do you often feel ?y?"
    ],
    "?*x I felt ?*y": [
        "What other feelings do you have?"
    ]
}

default_words = [
    "Sorry, I don't understand you.",
    "Please go on.",
]

test_inputs = ['hello there',
               'i want to test this program',
               'i could see if it works',
               'no not really',
               'no',
               'forget it-- i was wondering how general the program is',
               'i felt like it',
               'i feel this is enough',
               '[Abort]']

rules_list = []


def test_case():
    for w in test_inputs:
        if w.upper() == '[ABORT]':
            break
        print('I: ', w)
        print('ELIZA > ', use_eliza_rules(rules_list, w.upper(), default_words))


def main():
    for (pattern, matches) in eliza_rules.items():
        pattern = pattern.upper()
        matches = [t.upper() for t in matches]
        rules_list.append((pattern, matches))
    eliza(rules_list, list(map(str.upper, default_words)))


if __name__ == "__main__":
    main()

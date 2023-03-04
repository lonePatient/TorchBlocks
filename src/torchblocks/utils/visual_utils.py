import matplotlib.pylab as plt
from itertools import cycle

plt.style.use("ggplot")
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


def text_to_color(text, color):
    """
    Returns highlighted text for the string
    """
    return f"\033[4{color};30m{text}\033[m"


def get_colored_sentence(sentence, label_to_color):
    colored_sentence = []
    spans = sentence['entities']
    text = sentence['text']
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    span_num = len(spans)
    b_end = 0
    for i in range(span_num):
        tag, start, end, fragment = spans[i]
        if start > b_end:
            O_fragment = text[b_end:start]
            color = label_to_color['None']
            colored_text = text_to_color(O_fragment, color)
            colored_sentence.append(colored_text)
        color = label_to_color[tag]
        colored_text = text_to_color(fragment, color)
        colored_sentence.append(colored_text)
        b_end = end
    return " ".join(colored_sentence)


if __name__ == "__main__":
    sentence = {'id': '0',
                'text': '大于三十岁的与临时居住地是松陵村东区473号楼的初婚东乡族同网吧的学历为高中的外来务工人员',
                'entities': [['AGE', 0, 5, '大于三十岁'],
                             ['EDU', 36, 38, '高中'],
                             ['TAG', 39, 45, '外来务工人员'],
                             ['PER', 13, 23, '松陵村东区473号楼']],
                'intent': 'KBQA'
                }
    label_to_color = {
        "AGE": 1,  # 1 red
        "EDU": 2,  # 2 green
        "TAG": 3,  # 3 yellow
        "PER": 4,  # 4 blue
        "None": 9,  # default
    }
    # jupyter notebook
    # FLAT NER
    print(get_colored_sentence(sentence, label_to_color))

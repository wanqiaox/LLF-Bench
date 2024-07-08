import random
import re
from string import punctuation

import gym
import cmudict
import syllables
import sys
import string

from gym.utils import seeding

from llfbench.utils.parser_utils import SimpleGuidanceParser
from llfbench.envs.llf_env import Feedback


class PoemUtil:
    # designed as a Mixin class
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # forwards all unused arguments
        self.cmudict = cmudict.dict()

    def simple_syllable_count(self, word):
        # can also use pip syllables library
        text = word.lower()
        # remove non-alphabets
        text = re.sub('[^a-z]', '', text)

        # count syllables based on phonetics rules
        count = 0
        vowels = 'aeiouy'
        if len(text) == 0:
            return count
        if text[0] in vowels:
            count += 1
        for index in range(1, len(text)):
            if text[index] in vowels and text[index - 1] not in vowels:
                count += 1
        if text.endswith('e'):
            count -= 1
        if text.endswith('le') and text[-3] not in vowels:
            count += 1
        if text.endswith('ed'):
            count -= 1
        if count == 0:
            count += 1
        return count

    def count_syllables(self, line):
        """Use corpora to count syllables in English word or phrase."""
        # prep words for cmudict corpus
        line = line.replace('-', ' ')
        words = line.lower().split()
        num_sylls = 0
        for word in words:
            word = word.strip(punctuation)
            if word.endswith("'s") or word.endswith("’s"):
                word = word[:-2]
            # if word in missing_words:
            #     num_sylls += missing_words[word]
            result = self.cmudict[word]
            # if there is no result, we try to do a simple count
            if len(result) == 0:
                # heuristic based checking
                num_sylls += syllables.estimate(word)  # simple_syllable_count(word)
                continue

            for phonemes in self.cmudict[word][0]:
                for phoneme in phonemes:
                    if phoneme[-1].isdigit():
                        num_sylls += 1
        return num_sylls

    def seed(self, seed):
        pass


class PoemExtractor(object):
    # use LLM to extract the poem
    # just in case more things were written
    def __init__(self, llm, silent=True):
        self.llm = llm
        self.prompt = SimpleGuidanceParser("""
{{#system~}}
You are a helpful assistant.
{{~/system}}

{{#user~}}
Extract only lines of poems in the following message, ignore any part of the message that is not related to the poem.
Only return the poem line by line, including space.

```
{{content}}
```
{{~/user}}

{{#assistant~}}
{{gen 'poem' temperature=0.7}}
{{~/assistant}}
""")

    def __call__(self, content):
        messages = self.prompt(content=content)
        response, info = self.llm.generate(messages)
        return response


class Haiku(PoemUtil, gym.Env):
    def __init__(self, feedback=0, use_extractor=False, seed=None):
        self.assignment = f"Can you write me a haiku? A haiku is a poem that consists of three phrases composed of 17 syllables in a 5, 7, 5 pattern."
        self.form_name = 'Haiku'
        self.use_extractor = use_extractor
        self.extractor = None

        self.feedback = feedback
        self.syllable_req = [5, 7, 5]
        self.syllable_req_str = [str(i) for i in self.syllable_req]
        assert feedback in {0, 0.5, 1}

        self.action_space = gym.spaces.Text(sys.maxsize, charset=string.printable)
        self.observation_space = gym.spaces.Text(sys.maxsize, charset=string.printable)

        self._seed = self.seed(seed)

        self.docstring = self.assignment

        super().__init__()

    def reset(self, **kwargs):
        return self.assignment

    def seed(self, seed=None):
        """Seed the PRNG of this space and possibly the PRNGs of subspaces."""
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def initialize_text_extractor(self, poem_extractor: PoemExtractor):
        self.extractor = poem_extractor

    def line_number_incorrect(self, observed_num):
        # The line number is incorrect.
        assert observed_num != len(self.syllable_req)

        improv_direction = "more" if observed_num < len(self.syllable_req) else "less"

        didactic_feedback = Feedback()
        didactic_feedback.r = f"The generated {self.form_name} is incorrect."
        didactic_feedback.fp = f"Write a {self.form_name} that has exactly {len(self.syllable_req)} lines. Write {improv_direction} lines."
        didactic_feedback.fn = f"Do not write a {self.form_name} that has more or less lines than {len(self.syllable_req)}."
        didactic_feedback.hn = f"You wrote {observed_num} lines but the poem needs to be have exactly {len(self.syllable_req)} lines."

        if self.feedback == 0:
            feedback = f"The generated {self.form_name} is incorrect."
        elif self.feedback == 0.5:
            feedback = f"The generated {self.form_name} is incorrect. This is because the {self.form_name} needs to have exactly {len(self.syllable_req)} lines."
        elif self.feedback == 1:
            improv_direction = "more" if observed_num < len(self.syllable_req) else "less"
            feedback = f"The generated {self.form_name} is incorrect. This is because the {self.form_name} needs to have exactly {len(self.syllable_req)} lines. You wrote {observed_num} lines. Write {improv_direction} lines."
        else:
            raise ValueError(f"Invalid feedback level: {self.feedback}")

        return feedback, didactic_feedback

    def line_syllable_check(self, lines):
        success = True
        success_line, total_line = 0, 0
        error_info, success_info = [], []

        for i in range(len(self.syllable_req)):
            # this is to say -- if the generated poem is shorter than required lines
            # we just count the missing line as wrong (for the missing line)
            if i >= len(lines):
                success = False
                total_line += 1
                continue

            line = lines[i]
            count = self.count_syllables(line)
            success *= count == self.syllable_req[i]
            if count != self.syllable_req[i]:
                diff = self.syllable_req[i] - count  # positive: increase syllable; negative: decrease syllable
                error_info.append([i, line, count, diff])
            else:
                success_line += 1
                success_info.append([i, line, count, 0])
            total_line += 1

        return success, success_line / total_line, error_info, success_info

    def produce_line_feedback(self, error_info, success_info):
        # This is called when the line number is correct
        # produce didactic feedback
        didactic_feedback = Feedback()
        if len(error_info) == 0:  # success
            # this is the only place sucess feedback is produced
            didactic_feedback.r = f"The generated {self.form_name} is correct. Congrats! You have successfully produced a poem that matches the assignment description."
            feedback = didactic_feedback.r
            return feedback, didactic_feedback
        else:
            didactic_feedback.r = f"The generated {self.form_name} is incorrect."
            didactic_feedback.hn = f"{self.form_name} needs to have exactly {'-'.join(self.syllable_req_str)} syllables in {len(self.syllable_req)} lines"
            didactic_feedback.hn += ", but lines " if len(error_info) > 1 else ", but line "
            for tup in error_info:
                i, line, count, diff = tup
                # feedback +=  f'The line: "{line}" has {count} syllables. It should only have {self.syllable} syllables' + '\n'
                didactic_feedback.hn += f"{i + 1},"
            didactic_feedback.hn = didactic_feedback.hn[:-1]
            didactic_feedback.hn += " do not." if len(error_info) > 1 else " does not."

            didactic_feedback.hp = "These lines are correct because they have the correct syllables: "
            for tup in success_info:
                i, line, count, diff = tup
                didactic_feedback.hp += f"line {i + 1} has {count} syllables,"
            didactic_feedback.hp = didactic_feedback.hp[:-1]
            didactic_feedback.hp += "."

            didactic_feedback.fp = "Here are some suggestions to fix your error:\n"
            for tup in error_info:
                i, line, count, diff = tup
                improv_direction = "more" if diff > 0 else "less"
                didactic_feedback.fp += f'The line: "{line}" has {count} syllables. It should only have {self.syllable_req[i]} syllables. '
                didactic_feedback.fp += f'You should rewrite the line to have {improv_direction} syllables.' + '\n'

        # now we know there's an error
        if self.feedback == 0:
            # we just say "The generated poem is not correct."
            feedback = f"The generated {self.form_name} is incorrect."
        elif self.feedback == 0.5:
            # we offer an explanation or error message (on exactly which line is at fault)
            # Generated poem is incorrect because <which rule was violated, and where:> poem needs to have exactly 7 syllables in each line, but lines x,y do not.
            feedback = f"The generated {self.form_name} is incorrect.\n"
            feedback += f"This is because {self.form_name} needs to have exactly {'-'.join(self.syllable_req_str)} syllables in {len(self.syllable_req)} lines"
            feedback += ", but lines " if len(error_info) > 1 else ", but line "
            for tup in error_info:
                i, line, count, diff = tup
                feedback += f"{i + 1},"
            feedback = feedback[:-1]
            feedback += " do not." if len(error_info) > 1 else " does not."
        elif self.feedback == 1:
            feedback = f"The generated {self.form_name} is incorrect.\n"
            feedback += "Here are some suggestions to fix your error:\n"
            for tup in error_info:
                i, line, count, diff = tup
                improv_direction = "more" if diff > 0 else "less"
                feedback += f'The line: "{line}" has {count} syllables. It should only have {self.syllable_req[i]} syllables. '
                feedback += f'You should rewrite the line to have {improv_direction} syllables.' + '\n'

        return feedback, didactic_feedback

    def step(self, a):
        """
        LineSyllableConstrainedPoem environment has two latent attributes:
        - Did you have the right number of lines?
        - Did you have the right number of syllables in each line?

        The feedback is structured -- if the line number is wrong, we only provide feedback for the line number.
        (Because it's meaningless to compare syllables)

        If the line number is correct, we provide feedback for the syllables.
        """

        if self.use_extractor:
            if self.extractor is None:
                raise Exception(
                    "Must pass in an extractor through initialize_text_extractor before using the extractor.")
            a = self.extractor(a)

        feedbacks, didactic_feedback = [], Feedback()
        success = True

        lines = []
        for line in a.strip().split('\n'):
            if line == '':
                continue
            lines.append(line)

        if len(lines) != len(self.syllable_req):
            success = False

            feedback, didactic_feedback = self.line_number_incorrect(len(lines))
            feedbacks.append(feedback)
            frac = 0
            # if line numbers are wrong, it's a severe problem, reward we manually set to be 0
        else:
            syllabel_success, frac, error_info, success_info = self.line_syllable_check(lines)
            assert syllabel_success == (len(error_info) == 0)
            success *= syllabel_success

            feedback, didactic_feedback = self.produce_line_feedback(error_info, success_info)
            feedbacks.append(feedback)

        terminal = False  # one step environment

        if type(success) == int:
            success = success == 1

        # observation, reward, terminated, info
        return self.assignment, frac, terminal, {'original_feedback': feedback,
                                                 'feedback': didactic_feedback,
                                                 'success': success}


class Tanka(Haiku):
    def __init__(self, feedback=0, use_extractor=False, seed=None):
        # We can extend this to add "theme" of the poem
        # This increases difficulty a little, but also hard to check if it's thematic or not.
        # feedback=0, use_extractor=False, seed=None
        super().__init__(feedback, use_extractor, seed=seed)
        self.assignment = f"Can you write me a Tanka? A Tanka is a poem that consists of five lines composed of syllables in a 5-7-5-7-7 pattern."
        self.use_extractor = use_extractor
        self.feedback = feedback
        self.syllable_req = [5, 7, 5, 7, 7]
        self.syllable_req_str = [str(i) for i in self.syllable_req]
        self.form_name = 'Tanka'

        self.docstring = self.assignment


class LineSyllableConstrainedPoem(Haiku):
    def __init__(self, syllable_req=[7, 7, 7], feedback=0, use_extractor=False,
                 seed=None):
        # We can extend this to add "theme" of the poem
        # This increases difficulty a little, but also hard to check if it's thematic or not.
        super().__init__(feedback, use_extractor, seed=seed)
        self.syllable_req_str = [str(i) for i in syllable_req]
        self.assignment = f"Can you write me a poem? It should have {len(syllable_req)} lines. The number of syllables for the lines in the poem should follow a {'-'.join(self.syllable_req_str)} pattern."
        self.use_extractor = use_extractor
        self.feedback = feedback
        self.syllable_req = syllable_req
        self.form_name = 'poem'

        self.docstring = self.assignment

        self._seed = self.seed(seed)

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            self._seed = self.seed(kwargs['seed'])

        # create a sampling space
        # Haiku: 3, Tanka: 5, Sonnet: 14, Villanelle: 19, Ballad: 4, Ghazal: 15
        number_of_lines = self._np_random.choice([3, 4, 5, 14, 15, 19])
        # https://www.writing.upenn.edu/~afilreis/88/meter.html
        syllable_sample_space = [5, 7, 8, 9, 10, 17]

        syllable_req = []
        for _ in range(number_of_lines):
            syllable_req.append(self._np_random.choice(syllable_sample_space))

        self.syllable_req_str = [str(i) for i in syllable_req]
        self.syllable_req = syllable_req
        self.assignment = f"Can you write me a poem? It should have {len(syllable_req)} lines. The number of syllables for the lines in the poem should follow a {'-'.join(self.syllable_req_str)} pattern."

        return self.assignment


class SyllableConstrainedPoem(PoemUtil, gym.Env):
    def __init__(self, syllable=7, feedback=0, use_extractor=False, seed=None):

        super().__init__()
        self.assignment = f"Can you produce a short poem where each line has {syllable} syllables?"
        self.syllable = syllable
        self.use_extractor = use_extractor

        self.feedback = feedback
        assert self.feedback in {0, 0.5, 1}

        self.cmudict = cmudict.dict()
        self.extractor = None

        self.action_space = gym.spaces.Text(sys.maxsize, charset=string.printable)
        self.observation_space = gym.spaces.Text(sys.maxsize, charset=string.printable)

        self._seed = self.seed(seed)

        self.docstring = self.assignment

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            self._seed = self.seed(kwargs['seed'])

        # https://www.writing.upenn.edu/~afilreis/88/meter.html
        syllable_sample_space = [5, 7, 8, 9, 10, 17]
        syllable = self._np_random.choice(syllable_sample_space)
        self.syllable = syllable
        self.assignment = f"Can you produce a short poem where each line has {syllable} syllables?"

        return self.assignment

    def seed(self, seed=None):
        """Seed the PRNG of this space and possibly the PRNGs of subspaces."""
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def initialize_text_extractor(self, poem_extractor: PoemExtractor):
        self.extractor = poem_extractor

    def get_line_feedback(self, text):
        success = True
        success_line, total_line = 0, 0
        error_info, success_info = [], []
        for i, line in enumerate(text.strip().split('\n')):
            if line == '':
                # this means it's just a segment break
                continue
            count = self.count_syllables(line)
            success *= count == self.syllable
            if count != self.syllable:
                diff = self.syllable - count  # positive: increase syllable; negative: decrease syllable
                error_info.append([i, line, count, diff])
            else:
                success_line += 1
                success_info.append([i, line, count, 0])
            total_line += 1
        return success, success_line / total_line, error_info, success_info

    def step(self, a):
        # observation, reward, terminal, info
        if self.use_extractor:
            if self.extractor is None:
                raise Exception(
                    "Must pass in an extractor through initialize_text_extractor before using the extractor.")
            a = self.extractor(a)
        success, frac, error_info, success_info = self.get_line_feedback(a)

        if success:
            feedback = f"The generated poem is correct. Congrats! You have successfully produced a poem that matches the assignment description."
            didactic_feedback = Feedback(r=feedback)
            return self.assignment, frac, False, {'frac': frac, 'original_feedback': feedback,
                                                  'feedback': didactic_feedback, 'success': True}

        if self.feedback == 0:
            feedback = "The generated poem is incorrect."
        elif self.feedback == 0.5:
            # we offer an explanation or error message (on exactly which line is at fault)
            # Generated poem is incorrect because <which rule was violated, and where> poem needs to have exactly 7 syllables in each line, but lines x,y do not.
            feedback = "The generated poem is incorrect.\n"
            feedback += f"This is because the poem needs to have exactly {self.syllable} syllables in each line"
            feedback += ", but lines " if len(error_info) > 1 else ", but line "
            for tup in error_info:
                i, line, count, diff = tup
                feedback += f"{i + 1},"
            feedback = feedback[:-1]
            feedback += " do not." if len(error_info) > 1 else " does not."
            feedback = "The generated poem is incorrect.\n"
            feedback += "Here are some suggestions to fix your error:\n"
            for tup in error_info:
                i, line, count, diff = tup
                improv_direction = "more" if diff > 0 else "less"
                feedback += f'The line: "{line}" has {count} syllables. It should only have {self.syllable} syllables. '
                feedback += f'You should rewrite the line to have {improv_direction} syllables.' + '\n'
        else:
            raise ValueError(f"Invalid feedback level: {self.feedback}")

        terminal = False  # one step environment

        didactic_feedback = Feedback()
        didactic_feedback.r = "The generated poem is incorrect."

        didactic_feedback.hn = f"The poem needs to have exactly {self.syllable} syllables in all lines"
        didactic_feedback.hn += ", but lines " if len(error_info) > 1 else ", but line "
        for tup in error_info:
            i, line, count, diff = tup
            didactic_feedback.hn += f"{i + 1},"
        didactic_feedback.hn = didactic_feedback.hn[:-1]
        didactic_feedback.hn += " do not." if len(error_info) > 1 else " does not."

        didactic_feedback.hp = "These lines are correct because they have the correct syllables: "
        for tup in success_info:
            i, line, count, diff = tup
            didactic_feedback.hp += f"line {i + 1} has {count} syllables,"
        didactic_feedback.hp = didactic_feedback.hp[:-1]
        didactic_feedback.hp += "."

        didactic_feedback.fp = "Here are some suggestions to fix your error:\n"
        for tup in error_info:
            i, line, count, diff = tup
            improv_direction = "more" if diff > 0 else "less"
            didactic_feedback.fp += f'The line: "{line}" has {count} syllables. It should only have {self.syllable} syllables. '
            didactic_feedback.fp += f'You should rewrite the line to have {improv_direction} syllables.' + '\n'

        out = self.assignment, frac, terminal, {'frac': frac, 'original_feedback': feedback,
                                                "feedback": didactic_feedback, 'success': False}
        return out


class HierarchicalLineSyllableConstrainedPoem(PoemUtil, gym.Env):
    '''
    This environment is designed to be hierarchical. 
    It first determines a total number of lines to check. 
    If the number of generated lines is larger than the lines to check, it goes on to check the syllables in each line.
    Then, for each of these lines, it checks if the syllabus in each line is correct. 
    By 'correct', we mean that the number of syllables in each line should lie on one side of a threshold. 
    When the side is 1, the number of syllables should be larger than the threshold. 
    This threshold is determined by the [syllable_thres] parameter.
    The side of the threshold is determined by the [side] parameter.
    '''
    def __init__(self, syllable_thres=[7, 7, 7], side=[1, 0, 1], 
                 context=0, feedback=0, use_extractor=False,
                 seed=None):
        super().__init__()
        assert context <= 1 and context >= 0
        assert feedback <= 1 and feedback >= 0
        assert len(syllable_thres) == len(side)
        self.syllable_req_str = [str(i) for i in syllable_thres]
        self.assignment = f"Can you write me a poem?"
        if context > 0:
            length_context = f" It should have at least {len(syllable_thres)} lines."
            self.assignment += length_context
            if context > 1:
                self.assignment += f" The number of syllables for the first {context - 1} lines in the poem should obey the following rules."
                for i in range(context - 1):
                    self.assignment += f" Line number {i + 1} should have at least {syllable_thres[i]} syllables." if side[i] == 1 else f" Line number {i + 1} should have less than {syllable_thres[i]} syllabus."
        self.use_extractor = use_extractor
        self.feedback = feedback
        self.syllable_thres = syllable_thres
        self.side = side
        self.context = context
        self.correctness = [False] * (len(syllable_thres) + 1)
        self.form_name = 'poem'

        self.docstring = self.assignment

        self.action_space = gym.spaces.Text(sys.maxsize, charset=string.printable)
        self.observation_space = gym.spaces.Text(sys.maxsize, charset=string.printable)

        self._seed = self.seed(seed)

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            self._seed = self.seed(kwargs['seed'])
        # create a sampling space
        # Haiku: 3, Tanka: 5, Sonnet: 14, Villanelle: 19, Ballad: 4, Ghazal: 15
        number_of_lines = self._np_random.choice([5])
        # https://www.writing.upenn.edu/~afilreis/88/meter.html
        syllable_sample_space = [3, 5, 7, 8, 9, 10, 15]
        
        syllable_thres = []
        for _ in range(number_of_lines):
            syllable_thres.append(self._np_random.choice(syllable_sample_space))
        side = []
        for _ in range(number_of_lines):
            side.append(self._np_random.choice([0, 1]))
        
        self.syllable_req_str = [str(i) for i in syllable_thres]
        self.syllable_thres = syllable_thres
        self.side = side
        self.assignment = f"Can you write me a poem?"
        self.correctness = [False] * (len(syllable_thres) + 1)

        import math
        context_upto = min(math.ceil(self.context * len(self.syllable_thres)), len(self.syllable_thres)) # feedback up to this line
        if self.context > 0:
            length_context = f" It should have at least {len(self.syllable_thres)} lines. The first {len(self.syllable_thres)} lines will be checked in order."
            self.assignment += length_context + f" Correctness of each line will be determined by the number of syllables it contains."
            if context_upto >= 1:
                self.assignment += f" The number of syllables for the first {context_upto} "
                self.assignment += f"lines" if self.context > 1 else f"line"
                self.assignment += f" in the poem should obey the following rules."
                for i in range(self.context - 1):
                    self.assignment += f" Line number {i + 1} should have at least {self.syllable_thres[i]} syllables." if self.side[i] == 1 else f" Line number {i + 1} should have less than {self.syllable_thres[i]} syllabus."
        
        return self.assignment
    
    def seed(self, seed=None):
        """Seed the PRNG of this space and possibly the PRNGs of subspaces."""
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def initialize_text_extractor(self, poem_extractor: PoemExtractor):
        self.extractor = poem_extractor

    def check_length(self, text):
        lines = []
        for line in text.strip().split('\n'):
            if line == '':
                continue
            lines.append(line)
        self.correctness[0] = (len(lines) >= len(self.syllable_thres))
        return lines
    
    def check_syllables(self, line_id, lines):
        assert line_id < len(self.syllable_thres)
        # Track correctness internally, without revealing any other info to the agent except for the feedback. 
        if False in self.correctness[:line_id + 1]:
            self.correctness[line_id + 1] = (self._np_random.integers(low=0, high=2) == 0)
        else:
            line = lines[line_id]
            s = self.count_syllables(line)
            if self.side[line_id] == 0:
                self.correctness[line_id + 1] = (s <= self.syllable_thres[line_id])
            else:
                self.correctness[line_id + 1] = (s > self.syllable_thres[line_id])
        return lines
    
    def line_number_incorrect(self, observed_num):
        # The line number is incorrect.
        assert observed_num < len(self.syllable_thres)

        didactic_feedback = Feedback()
        # If feedback is 0, we only say the poem is incorrect.
        didactic_feedback.r = f"The generated {self.form_name} is incorrect."
        if self.feedback == 0:
            didactic_feedback.fp = f"The generated {self.form_name} is incorrect."
            didactic_feedback.fn = f"The generated {self.form_name} is incorrect."
            didactic_feedback.hn = f"The generated {self.form_name} is incorrect."
        else:
            didactic_feedback.fp = f"Write a {self.form_name} that has at least {len(self.syllable_thres)} lines. Write more lines."
            didactic_feedback.fn = f"Do not write a {self.form_name} that has less lines than {len(self.syllable_thres)}."
            didactic_feedback.hn = f"You wrote {observed_num} lines but the poem needs to be have at least {len(self.syllable_thres)} lines."

        if self.feedback == 0:
            feedback = f"The generated {self.form_name} is incorrect."
        else:
            feedback = f"The generated {self.form_name} is incorrect. This is because the {self.form_name} needs to have at least {len(self.syllable_thres)} lines. You wrote {observed_num} lines. Write more lines."
            
        return feedback, didactic_feedback
    
    def line_syllable_check(self, checks, lines):
        success = True
        success_line, total_line = 0, 0
        error_info, success_info = [], []
        for i, check in enumerate(checks[1:]):
            if False in checks[:i + 1]:
                assert success == False
                total_line += 1
                continue
            if not check:
                error_info.append([i, lines[i], self.count_syllables(lines[i])])
            else:
                success_line += 1
                success_info.append([i, lines[i], self.count_syllables(lines[i])])
            success *= check
            total_line += 1

        return success, success_line / total_line, error_info, success_info
        
    def produce_line_feedback(self, error_info, success_info):
        # TODO: change feedback to encourage LM to only turn one knob at a time (follow the curriculum)
        import math
        feedback_upto = min(math.ceil(self.feedback * len(self.syllable_thres)), len(self.syllable_thres)) # feedback up to this line
        # This is called when the line number is correct
        improv_direction = ["less", "more"]
        # produce didactic feedback
        didactic_feedback = Feedback()
        if len(error_info) == 0:  # success
            # this is the only place sucess feedback is produced
            didactic_feedback.r = f"The generated {self.form_name} is correct. Congrats! You have successfully produced a poem that matches the assignment description."
            feedback = didactic_feedback.r
            return feedback, didactic_feedback
        else:
            didactic_feedback.r = f"The generated {self.form_name} is incorrect."
            if feedback_upto > 0:
                didactic_feedback.hn = f" The number of syllables for the first {feedback_upto} lines in the poem should obey the following rules."
            for i in range(feedback_upto):
                didactic_feedback.hn += f" Line number {i + 1} should have at least {self.syllable_thres[i]} syllables." if self.side[i] == 1 else f" Line number {i + 1} should have less than {self.syllable_thres[i]} syllabus."
            if feedback_upto > 0:
                didactic_feedback.hn += " Yet lines " if sum([error[0] < feedback_upto for error in error_info]) > 1 else " Yet line "
                for tup in error_info:
                    i, line, count = tup
                    if i < feedback_upto:
                        didactic_feedback.hn += f"{i + 1},"
                didactic_feedback.hn = didactic_feedback.hn[:-1]
                didactic_feedback.hn += " do not." if sum([error[0] < feedback_upto for error in error_info]) > 1 else " does not."

            if feedback_upto > 0 and len(success_info) > 0:
                didactic_feedback.hp = "These lines are correct because they have the correct syllables: "
                for tup in success_info:
                    i, line, count = tup
                    if i < feedback_upto:
                        side = self.side[i]
                        didactic_feedback.hp += f" line {i + 1} has {count} syllables, {improv_direction[side]} than {self.syllable_thres[i]} syllabus,"
                didactic_feedback.hp = didactic_feedback.hp[:-1]
                didactic_feedback.hp += "."

            if feedback_upto > 0:
                didactic_feedback.fp = "Here are some suggestions to fix your error:\n"
                for tup in error_info:
                    i, line, count = tup
                    if i < feedback_upto:
                        side = self.side[i]
                        didactic_feedback.fp += f'Line number {i + 1} has {count} syllables. It should have {improv_direction[side]} than {self.syllable_thres[i]} syllables. '
                        didactic_feedback.fp += f'You should rewrite the line to have {improv_direction[side]} syllables.' + '\n'

        # now we know there's an error
        if feedback_upto == 0:
            # we just say "The generated poem is not correct."
            feedback = f"The generated {self.form_name} is incorrect."
        else:
            # we offer feedback up to the feedback_line
            # we offer an explanation or error message (on exactly which line is at fault) before feedback_line
            feedback = f"The generated {self.form_name} is incorrect.\n"
            feedback += f" The number of syllables for the first {feedback_upto} lines in the poem should obey the following rules."
            for i in range(feedback_upto):
                feedback += f" Line number {i + 1} should have at least {self.syllable_thres[i]} syllables." if self.side[i] == 1 else f" Line number {i + 1} should have less than {self.syllable_thres[i]} syllabus."
            feedback += ", but lines " if sum([error[0] < feedback_upto for error in error_info]) > 1 else ", but line "
            for tup in error_info:
                i, line, count = tup
                if i < feedback_upto:
                    feedback += f"{i + 1},"
            feedback = feedback[:-1]
            feedback += " do not." if sum([error[0] < feedback_upto for error in error_info]) > 1 else " does not."

            feedback += "Here are some suggestions to fix your error:\n"
            for tup in error_info:
                i, line, count = tup
                if i < feedback_upto:
                    side = self.side[i]
                    feedback += f'Line number {i + 1} has {count} syllables. It should have {improv_direction[side]} than {self.syllable_thres[i]} syllables. '
                    feedback += f'You should rewrite the line to have {improv_direction} syllables.' + '\n'

        return feedback, didactic_feedback

    def step(self, lines):
        checks = self.correctness
        assert len(checks) == len(self.syllable_thres) + 1
        feedbacks, didactic_feedback = [], Feedback()
        success = True
        if not checks[0]:
            success = False
            feedback, didactic_feedback = self.line_number_incorrect(len(lines))
            feedbacks.append(feedback)
            frac = 0
            # if line numbers are not enough, it's a first problem, reward we manually set to be 0
        else:
            syllable_success, frac, error_info, success_info = self.line_syllable_check(checks, lines)
            assert syllable_success == (len(error_info) == 0)
            success *= syllable_success
            feedback, didactic_feedback = self.produce_line_feedback(error_info, success_info)
            feedbacks.append(feedback)

        terminal = False  # one step environment

        if type(success) == int:
            success = success == 1

        # observation, reward, terminated, info
        return self.assignment, frac, terminal, {'original_feedback': feedback,
                                                 'feedback': didactic_feedback,
                                                 'success': success}
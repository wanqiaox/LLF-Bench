from typing import SupportsFloat
from llfbench.envs.env_wrappers import TerminalFreeWrapper, EnvCompatibility
from llfbench.envs.llf_env import LLFWrapper, Feedback
from llfbench.envs.poem.formal_poems import Haiku, Tanka, LineSyllableConstrainedPoem, SyllableConstrainedPoem, HierarchicalLineSyllableConstrainedPoem, NumericalPlanningPoem
from llfbench.envs.poem.prompts import *

class PoemGymWrapper(LLFWrapper):

    INSTRUCTION_TYPES = ('b')
    FEEDBACK_TYPES = ('r', 'hp', 'hn', 'fp', 'fn')

    def __init__(self, env, instruction_type, feedback_type):
        super().__init__(TerminalFreeWrapper(EnvCompatibility(env)), instruction_type, feedback_type)

    @property
    def reward_range(self):
        return (-1.0, 0.0)

    def _reset(self, *, seed=None, options=None):  # TODO types of instructions
        instruction, info = self.env.reset(seed=seed, options=options)
        info['success'] = False
        if type(self._poem_env) == Haiku:
            instruction = self.reformat(instruction, haiku_b_instruction)
        elif type(self._poem_env) == Tanka:
            instruction = self.reformat(instruction, tanka_b_instruction)
        elif type(self._poem_env) == LineSyllableConstrainedPoem:
            instruction = self.reformat(instruction, line_syllable_constrained_poem_b_instruction)
            if options is not None and 'syllable_req' in options:
                self.env.env.env.syllable_req = options['syllable_req']
                self.env.env.env.syllable_req_str = [str(i) for i in self.env.syllable_req]
                self.env.env.env.assignment = f"Can you write me a poem? It should have {len(self.env.syllable_req)} lines. The number of syllables for the lines in the poem should follow a {'-'.join(self.env.syllable_req_str)} pattern."
                instruction = self.reformat(self.env.env.env.assignment, line_syllable_constrained_poem_b_instruction)
        elif type(self._poem_env) == SyllableConstrainedPoem:
            instruction = self.reformat(instruction, syllable_constrained_poem_b_instruction)
            if options is not None and 'syllable' in options:
                self.env.env.env.syllable = options['syllable']
                self.env.env.env.assignment = f"Can you produce a short poem where each line has {self.env.syllable} syllables?"
                instruction = self.reformat(self.env.env.env.assignment, syllable_constrained_poem_b_instruction)
        elif type(self._poem_env) == HierarchicalLineSyllableConstrainedPoem:
            # TODO: reformat instructions and update self.env.assignment
            if options is not None and 'syllable_thres' in options:
                self.env.env.env.syllable_thres = options['syllable_thres']
                side = []
                for _ in range(len(self.env.env.env.syllable_thres)):
                    side.append(self.np_random.choice([0, 1]))
                self.env.env.env.side = side
            if options is not None and 'side' in options:
                assert len(options['side']) == len(self.env.env.env.syllable_thres)
                self.env.env.env.side = options['side']
            if options is not None and 'context' in options:
                self.env.env.env.context = options['context']
            if options is not None and 'feedback' in options:
                self.env.env.env.feedback = options['feedback']
        elif type(self._poem_env) == NumericalPlanningPoem:
            # TODO: paraphrase
            if options is not None and 'syllable_req' in options:
                self.env.env.env.req = options['syllable_req']
                self.env.env.env.checking_type = 'syllables'
            elif options is not None and 'word_req' in options:
                self.env.env.env.req = options['word_req']
                self.env.env.env.checking_type = 'words'
            if options is not None and 'starts_with' in options:
                self.env.env.env.starts_with = options['starts_with']
            if options is not None and 'ends_with' in options:
                self.env.env.env.ends_with = options['ends_with']
            if options is not None and 'context' in options:
                self.env.env.env.context = options['context']
            if options is not None and 'feedback' in options:
                self.env.env.env.feedback = options['feedback']
            self.env.env.env.checking_type = "syllables"
            self.env.env.env.para_req_str = [str(len(para)) for para in self.env.env.env.req]
            req_str = [[str(i) for i in para] for para in self.env.env.env.req]
            self.env.env.env.req_str = ["-".join(req_str[i]) for i in range(len(req_str))]
            self.env.env.env.assignment = f"Can you write me a poem?"
            if self.env.env.env.starts_with is not None and self.env.env.env.context > 0:
                self.env.env.env.assignment = f"Can you complete a poem '{self.env.env.env.starts_with}'?"
            if self.env.env.env.ends_with is not None and self.env.context > 0:
                if len(self.env.env.env.ends_with.split(' ')) == 1:
                    self.env.env.env.assignment = self.env.env.env.assignment.strip('?') + f" including the last word as {self.env.env.env.ends_with}?"
                else:
                    self.env.env.env.assignment = self.env.env.env.assignment.strip('?') + f" including the last sentence as {self.env.env.env.ends_with}?"
            para_context = f" It should have exactly {len(self.env.env.env.para_req_str)} "
            para_context += "paragraphs." if len(self.env.env.env.para_req_str) > 1 else "paragraph."
            line_context = f" The number of lines in each paragraph should follow a {'-'.join(self.env.env.env.para_req_str)} pattern." if len(self.env.env.env.para_req_str) > 1 else f" The paragraph should contain exactly {len(req_str[0])} lines."
            syllable_word_context = f" The number of {self.env.env.env.checking_type} in each line should follow a {', '.join(self.env.env.env.req_str).strip(', ')} pattern."
            if self.env.env.env.context > 1:
                self.env.env.env.assignment += para_context
                if self.env.env.env.context > 2:
                    self.env.env.env.assignment += line_context
                    if self.env.env.env.context > 3:
                        self.env.env.env.assignment += syllable_word_context
            self.env.env.env.docstring = self.env.env.env.assignment
            instruction = self.env.env.env.assignment
        return dict(instruction=instruction, observation=None, feedback=None), info

    def _step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        reward -= 1.0  # so that early stopping due to success would give the right return
        didactic_feedback = info['feedback']
        del info['feedback']
        del info['original_feedback']

        paraphrased_feedback = Feedback()
        for feedback_type in self._feedback_type:
            feedback = didactic_feedback[feedback_type]
            if feedback_type == 'r':
                feedback = self.reformat(feedback, r_feedback_pos)
                feedback = self.reformat(feedback, r_feedback_neg)
            elif feedback_type == 'hn':
                feedback = self.reformat(feedback, line_number_hn_feedback)
                feedback = self.reformat(feedback, syllable_hn_feedback)
            elif feedback_type == 'hp':
                feedback = self.reformat(feedback, syllable_hp_feedback)
            elif feedback_type == 'fp':
                feedback = self.reformat(feedback, line_number_fp_feedback)
                feedback = self.reformat(feedback, syllable_fp_feedback_1)
                feedback = self.reformat(feedback, syllable_fp_feedback_2)
            elif feedback_type == 'fn':
                feedback = self.reformat(feedback, line_number_fn_feedback)
            else:
                raise ValueError(f'Unknown feedback type: {feedback_type}')
            paraphrased_feedback[feedback_type] = feedback

        observation = dict(instruction=None, observation=None, feedback=paraphrased_feedback)

        return observation, reward, terminated, truncated, info

    def fix_sentence_capitalization(self, sentence):
        sentences = sentence.split(". ")
        fixed_sentences = [s[0].capitalize() + s[1:] if s else '' for s in sentences]
        fixed_sentence = ". ".join(fixed_sentences)
        return fixed_sentence

    def _verbalize_feedback(self, feedback_dict: Feedback) -> str:
        """ Implement this in the subclass to get the desired feedback string.
        """

        feedback = []
        for k, v in feedback_dict.asdict().items():
            if v is not None:
                line = self.fix_sentence_capitalization(f'{str(v)}')
                feedback.append(line)
        return ' '.join(feedback)

    @property
    def _poem_env(self):
        return self.env.env.env
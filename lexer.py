import argparse
#from pygments.lexer import RegexLexer
#from pygments.token import *

#numeric=['0','1','2','3','4','5','6','7','8','9']
alpha   = bytes(b'ABCDEFGHIJKLMNOPQRSTUUVWXYZabcdefghijklmnopqrstuvwxyz')	#\w - \d
numeric = bytes(b'0123456789')  	#\d
delim   = bytes(b' \t\r\n\f_-') 	#\s
eol 	= bytes(b'\r\n\f')

TEST_STRINGS=[
#	'word1 - suffix',
	'1prefix_word13',
	'22prefix_abcd24',
	'333prefix_xyzd35',
	'word1 - suffix',
	'word2 --      suffix',
	'word3     ---      suffix',
]

kUseRE=False 		#re can't handle some of the chars in 'delim'
if kUseRE:
	import re

def emit(prevstate, token):
	if token != '':
		print("  emit '%s'" % token)
	return token 	#TODO: change to a generator later for efficiency

def eat(prevstate, token):
	if token != '':
		print("  eat '%s'" % token)
	return token 	#TODO: change to a generator later for efficiency

class Lexer(object):
	def __init__(self, char_classes, class_labels):
		assert(len(char_classes) == len(class_labels))
		self._char_classes = char_classes
		self._class_labels = class_labels

		#1: lookup table that map a character to its class/label
		chardispatch = bytearray(256)
		self._chardispatch = chardispatch

		#2: build lookup table based on user's input - defines the rules for a token
		for idx, char_class in enumerate(char_classes):
			for char in char_class:
				chardispatch[char] = class_labels[idx][0]
			#delimiters are special
			if self.char_label_str(idx) == 'delim':
				self._delimiters = char_class
				self._delimiters4re = "[{}]".format(char_class)  #ready for re.split

		#use 2d array for clarify - use a single table for efficiency later
		# [(0, 'alpha'), (1, 'numeric'), (2, 'delim')]
		states = [
			#0: the initial root state (0)
			[1,    2, (emit, 3)],		
			#1: seen alpha state
			[1,    1, (emit, 3)],		#numeric following or within 'alpha' is part of the token
			#2: seen numeric state
			[2,    2,  (eat, 3)],		#alpha following numeric at the beginning is skipped as a token
			#3: emitted one or more tokens
			[3, (emit, 3), (emit, 3)],		#numeric following 'alpha' is NOT part of the token if it is not in state 0 (beginning)
			#4: FINAL (eol)
			#[0, eat,  eat],		#numeric following 'alpha' is NOT part of the token if it is not in state 0 (beginning)
		]
		
		self._states = states

	#lookup the character class label for 'char' (a numeric code)
	def get(self, char):
		return self._chardispatch[char]

	def char_label_str(self, char_label):
		return self._class_labels[char_label][1]

	@property
	def delimiters(self):
		return self._delimiters

	@property
	def delimiters4re(self):
		return self._delimiters4re

	@property
	def states(self):
		return self._states
	
	def next_state(self, state, char_label):
		return self.states[state][char_label]

	def do_eol(self, state, token):
		if token != '':
			if state == 0:
				emit(state, token)
			else:
				eat(state, token)

	def tokens(self, astring, kLogging=True):
		#1: use re.split to split along delimiters to take advantage of builtin speed.
		#   This could be handled entirely in the finite-state-automata also.
		#   It does assume delimiters are not sensitive to context. The FSA is much more flexible.
		if kUseRE:
			ourtokens = re.split(self.delimiters4re, astring)
			if kLogging:
				print("'%s': re.split %s" % (astring, ourtokens))
		else:
			if kLogging:
				print("'%s': " % (astring))
	
		state = 0 		#initial null state
		token = ''
		for char in astring:
			char_label = self.get(ord(char[0]))
			nextstate = self.next_state(state, char_label)

			if isinstance(nextstate, int):
				token += char
				state = nextstate
			else: 	#invoke action callback
				action = nextstate[0] 		#nextstate is now a tuple e.g. (emit, 3)
				token = action(state, token)
				state = nextstate[1]
				token = ''

		#simulate an EOL
		self.do_eol(state, token)

#class Lexer2(RegexLexer):


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='testlexer.py')
	parser.add_argument('--tests', type=str, default=TEST_STRINGS, help='test inputs')
	args = parser.parse_args()

	ourlexer = Lexer(
		[alpha, numeric, delim],	#sets of character grouped by their char class 
		[(0, 'alpha'), (1, 'numeric'), (2, 'delim')]	#(<char class>, <char class string>)
	)

	for char in alpha:
		char_label = ourlexer.get(char)
		#print("[%x] %d" % (char, char_label))

	for char in numeric:
		char_label = ourlexer.get(char)
		#print("[%x] %s" % (char, ourlexer.char_label_str(char_label)))

	for char in delim:
		char_label = ourlexer.get(char)
		#print("[%x] %s" % (char, ourlexer.char_label_str(char_label)))

	input = args.tests

	for test in TEST_STRINGS:
		tokens = ourlexer.tokens(test)



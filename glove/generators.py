import threading
class threadsafe_iter:
	"""Takes an iterator/generator and makes it thread-safe by
	serializing call to the `next` method of given iterator/generator.
	"""
	def __init__(self, it):
		self.it = it
		self.lock = threading.Lock()

	def __iter__(self):
		return self

	def __next__(self):
		with self.lock:
			return next(self.it)

def threadsafe_generator(f):
		"""A decorator that takes a generator function and makes it thread-safe.
		"""
		def g(*a, **kw):
			return threadsafe_iter(f(*a, **kw))
		return g

@threadsafe_generator
def baseline_data_generator(X, y, bs):
  [X_train_quest, X_train_doc] = X
  [y_start, y_end] = y
  len = X_train_quest.shape[0]
  i = 0

  while(True):
    X_train_quest_out = X_train_quest[i:i+bs]
    X_train_doc_out = X_train_doc[i:i+bs]
    y_start_out = y_start[i:i+bs]
    y_end_out = y_end[i:i+bs]

    X_out = [X_train_quest_out, X_train_doc_out]
    y_out = [y_start_out, y_end_out]

    yield(X_out, y_out)

    i+=bs

    if i >= len:
      i = 0

@threadsafe_generator
def features_data_generator(X, y, bs):
  [X_train_quest, X_train_doc, X_exact_lemmas, X_term_frequency] = X
  [y_start, y_end] = y
  len = X_train_quest.shape[0]
  i = 0

  while(True):
    X_train_quest_out = X_train_quest[i:i+bs]
    X_train_doc_out = X_train_doc[i:i+bs]
    X_exact_lemmas_out = X_exact_lemmas[i:i+bs]
    X_term_frequency_out = X_term_frequency[i:i+bs]
    y_start_out = y_start[i:i+bs]
    y_end_out = y_end[i:i+bs]

    X_out = [X_train_quest_out, X_train_doc_out, X_exact_lemmas_out, X_term_frequency_out]
    y_out = [y_start_out, y_end_out]

    yield(X_out, y_out)

    i+=bs

    if i >= len:
      i = 0
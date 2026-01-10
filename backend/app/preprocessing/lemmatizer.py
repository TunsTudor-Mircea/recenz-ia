"""
Lemmatization module for Romanian text preprocessing.

This module provides Romanian lemmatization using Stanza (Stanford NLP).
Stanza provides state-of-the-art Romanian NLP models including lemmatization.
"""

from typing import List, Set, Optional
import logging

logger = logging.getLogger(__name__)


# Romanian stopwords optimized for SENTIMENT ANALYSIS
# Preserves sentiment-critical words (negations, intensifiers, modals)
ROMANIAN_STOPWORDS_SENTIMENT = {
    'a', 'abia', 'acea', 'aceasta', 'această', 'aceea', 'acei', 'aceia',
    'acel', 'acela', 'acele', 'acelea', 'acest', 'acesta', 'aceste',
    'acestea', 'aceşti', 'aceştia', 'acolo', 'acord', 'acum', 'adica',
    'ai', 'aia', 'aibă', 'aici', 'aiurea', 'al', 'ala', 'alaturi',
    'ale', 'alea', 'alt', 'alta', 'altceva', 'altcineva', 'alte',
    'altfel', 'alti', 'altii', 'altul', 'am', 'anume', 'apoi', 'ar',
    'are', 'as', 'asa', 'asemenea', 'asta', 'astazi', 'astea', 'astfel',
    'astăzi', 'asupra', 'atare', 'atat', 'atata', 'atatea', 'atatia',
    'ati', 'atit', 'atita', 'atitea', 'atitia', 'atunci', 'au', 'avea',
    'avem', 'aveţi', 'avut', 'azi', 'aş', 'aşadar', 'aţi', 'b', 'ba',
    'c', 'ca', 'cam', 'cand', 'capat', 'care',
    'careia', 'carora', 'caruia', 'cat', 'catre', 'caut', 'ce', 'cea',
    'ceea', 'cei', 'ceilalti', 'cel', 'cele', 'celor', 'ceva', 'chiar',
    'ci', 'cinci', 'cind', 'cine', 'cineva', 'cit', 'cita', 'cite',
    'citeva', 'citi', 'citiva', 'conform', 'contra', 'cu', 'cui', 'cum',
    'cumva', 'curând', 'curînd', 'când', 'cât', 'câte', 'câtva', 'câţi',
    'd', 'da', 'daca', 'dacă', 'dar', 'dat', 'datorită', 'dată', 'dau',
    'de', 'deasupra', 'deci', 'decit', 'degraba', 'deja', 'deoarece',
    'departe', 'desi', 'despre', 'deşi', 'din', 'dinaintea', 'dintr',
    'dintr-', 'dintre', 'doi', 'doilea', 'două', 'drept', 'dupa',
    'după', 'dă', 'e', 'ea', 'ei', 'el', 'ele', 'eram', 'este', 'eu',
    'exact', 'eşti', 'f', 'face', 'fata', 'fel', 'fi', 'fie',
    'fiecare', 'fii', 'fim', 'fiu', 'fiţi', 'fost',
    'g', 'geaba', 'grație', 'h', 'halbă', 'i', 'ia', 'iar',
    'ieri', 'ii', 'il', 'imi', 'in', 'inainte', 'inapoi', 'inca',
    'incit', 'insa', 'intr', 'intre', 'isi', 'iti', 'j', 'k', 'l', 'la',
    'le', 'li', 'linga', 'lor', 'lui', 'lângă', 'lîngă', 'm', 'ma',
    'mare', 'mea', 'mei', 'mele', 'mereu', 'meu', 'mi', 'mie',
    'mine', 'mod', 'mulțumesc', 'mâine', 'mîine', 'mă', 'n', 'ne', 'nevoie', 'ni',
    'niste', 'nişte',
    'noastre', 'noastră', 'noi', 'nostri', 'nostru', 'nou',
    'nouă', 'noştri', 'număr', 'o', 'opt', 'or', 'ori', 'oricare',
    'orice', 'oricine', 'oricum', 'oricând', 'oricât', 'oricînd',
    'oricît', 'oriunde', 'p', 'pai', 'parca', 'patra', 'patru',
    'patrulea', 'pe', 'pentru', 'peste', 'pic', 'pina', 'plus',
    'prima', 'primul', 'prin', 'printr-', 'putini',
    'puţin', 'puţina', 'puţină', 'până', 'pînă', 'r', 'rog', 's', 'sa',
    'sa-mi', 'sa-ti', 'sai', 'sale', 'san', 'sunt', 'sută', 'sînt',
    'său', 't', 'ta', 'tale', 'tau', 'te', 'ti', 'timp', 'tine', 'toata',
    'toate', 'toată', 'tocmai', 'tot', 'toti', 'totul', 'totusi',
    'totuşi', 'toţi', 'trei', 'treia', 'treilea', 'tu', 'tuturor', 'tăi',
    'tău', 'u', 'ul', 'ului', 'un', 'una', 'unde', 'undeva', 'unei',
    'uneia', 'unele', 'uneori', 'uni', 'unii', 'unor', 'unora', 'unu',
    'unui', 'unuia', 'unul', 'v', 'va', 've', 'vei', 'voastre',
    'voastră', 'voi', 'vom', 'vor', 'vostru', 'vouă', 'voştri', 'vreme',
    'vreo', 'vreun', 'vă', 'x', 'z', 'zece', 'zero', 'zi', 'zice', 'îi',
    'îl', 'îmi', 'împotriva', 'în', 'înainte', 'înaintea', 'încotro',
    'încât', 'încît', 'între', 'întrucât', 'întrucît', 'îţi', 'ăla',
    'ălea', 'ăsta', 'ăstea', 'ăştia', 'şapte', 'şase', 'şi', 'ştiu',
    'ţi', 'ţie'
}


class RomanianLemmatizer:
    """
    Romanian lemmatizer using Stanza (Stanford NLP) with stopword removal.

    Stanza provides state-of-the-art Romanian lemmatization using neural models.
    Uses sentiment-optimized stopwords that preserve negations and intensifiers.

    Attributes:
        remove_stopwords: Whether to remove Romanian stopwords.
        stopwords: Set of Romanian stopwords to remove.
        _nlp: Stanza pipeline for Romanian lemmatization.
    """

    def __init__(self, remove_stopwords: bool = True):
        """
        Initialize RomanianLemmatizer.

        Args:
            remove_stopwords: If True, remove Romanian stopwords (sentiment-optimized).
        """
        self.remove_stopwords = remove_stopwords
        self.stopwords: Set[str] = ROMANIAN_STOPWORDS_SENTIMENT if remove_stopwords else set()
        self._nlp: Optional[any] = None
        self._model_downloaded = False

    def _ensure_lemmatizer(self) -> None:
        """Lazy load Stanza lemmatizer and download Romanian model if needed."""
        if self._nlp is None:
            try:
                import stanza

                # Check if model is downloaded, if not download it
                if not self._model_downloaded:
                    try:
                        logger.info("Downloading Stanza Romanian model (first time only)...")
                        stanza.download('ro', verbose=False)
                        self._model_downloaded = True
                        logger.info("Romanian model downloaded successfully")
                    except Exception as download_err:
                        logger.warning(f"Model download failed: {download_err}. Trying to use existing model...")

                # Initialize Stanza pipeline for Romanian
                # Use only lemmatization to save memory and speed
                logger.info("Loading Stanza Romanian lemmatizer...")

                # Try to use GPU if available, fall back to CPU
                import torch
                use_gpu = torch.cuda.is_available()
                if use_gpu:
                    logger.info("GPU detected - using GPU for Stanza lemmatization")
                else:
                    logger.info("No GPU detected - using CPU for Stanza lemmatization")

                self._nlp = stanza.Pipeline(
                    'ro',
                    processors='tokenize,lemma',
                    tokenize_pretokenized=True,  # We already have tokens
                    verbose=False,
                    use_gpu=use_gpu
                )
                logger.info("Stanza lemmatizer loaded successfully")

            except ImportError as e:
                logger.error("Stanza not installed. Please install with: pip install stanza")
                logger.warning("Lemmatization will be disabled. Only stopword removal will be performed.")
                self._nlp = None
            except Exception as e:
                logger.error(f"Failed to load Stanza lemmatizer: {e}")
                logger.warning("Lemmatization will be disabled. Only stopword removal will be performed.")
                self._nlp = None

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize a list of tokens using Stanza.

        Args:
            tokens: List of tokens to lemmatize.

        Returns:
            List of lemmatized tokens with stopwords optionally removed.
        """
        if not tokens:
            return []

        self._ensure_lemmatizer()

        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in self.stopwords]

        if not tokens:
            return []

        # If lemmatizer is not available, just return tokens
        if self._nlp is None:
            return tokens

        try:
            # Process with Stanza
            # Stanza expects text, but we set tokenize_pretokenized=True
            # so we need to pass tokens as a document
            text = ' '.join(tokens)
            doc = self._nlp(text)

            # Extract lemmas from the processed document
            lemmas = []
            for sentence in doc.sentences:
                for word in sentence.words:
                    lemmas.append(word.lemma)

            return lemmas

        except Exception as e:
            logger.warning(f"Lemmatization failed: {e}. Returning original tokens.")
            return tokens

    def lemmatize_batch(self, token_lists: List[List[str]]) -> List[List[str]]:
        """
        Lemmatize a batch of token lists.

        Args:
            token_lists: List of token lists to lemmatize.

        Returns:
            List of lemmatized token lists.
        """
        return [self.lemmatize(tokens) for tokens in token_lists]

    def __getstate__(self):
        """
        Prepare object for pickling.

        Exclude the Stanza pipeline as it cannot be pickled.
        It will be reloaded on demand when needed.
        """
        state = self.__dict__.copy()
        # Remove the unpicklable Stanza pipeline
        state['_nlp'] = None
        return state

    def __setstate__(self, state):
        """
        Restore object from pickle.

        The Stanza pipeline will be lazy-loaded when first used.
        """
        self.__dict__.update(state)
        # Pipeline will be reloaded on first use via _ensure_lemmatizer()

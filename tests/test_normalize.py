"""Name / address normalisation (src/normalize.py)."""
import polars as pl
import pytest

import normalize as N


class TestNameNormalisation:
    def test_devanagari_is_transliterated_and_flagged_nonlatin(self, norm):
        r = norm("भारत ग्रीन")
        assert r["name_core"] == "bhart grin"
        assert r["nonlatin"] == 1

    def test_latin_accents_are_folded_and_not_flagged(self, norm):
        r = norm("Café Étoile SARL")
        assert r["core_toks"] == ["cafe", "etoile"]
        assert r["nonlatin"] == 0

    def test_cjk_is_transliterated(self, norm):
        r = norm("东方 贸易")
        assert r["core_toks"] == ["dongfang", "maoyi"]
        assert r["nonlatin"] == 1

    @pytest.mark.parametrize("variant", ["Sharma Pvt. Ltd.", "Sharma Praivet Limited",
                                         "Sharma Praibhet Limitet", "Sharma Pte Ltd"])
    def test_legal_form_variants_collapse_to_one_canonical_form(self, norm, variant):
        r = norm(variant)
        assert r["legal"] == "limited private"
        assert r["core_toks"] == ["sharma"]

    def test_indic_script_legal_form_matches_latin_spelling(self, norm):
        r = norm("भारत ग्रीन प्राइवेट लिमिटेड")
        assert r["legal"] == "limited private"
        assert r["core_toks"] == norm("भारत ग्रीन")["core_toks"]

    def test_long_legal_words_are_shortened(self, norm):
        assert norm("Smith Incorporated")["legal"] == "inc"
        assert norm("Dupont et Fils Cie")["legal"] == "co"

    def test_stop_words_and_ampersand_are_dropped_from_core(self, norm):
        assert norm("The Bakery of Paris")["core_toks"] == ["bakery", "paris"]
        assert norm("Acme & Sons Co.")["core_toks"] == ["acme", "sons"]

    def test_name_made_only_of_legal_forms_keeps_its_tokens(self, norm):
        r = norm("Pvt Ltd")
        assert r["core_toks"] == ["private", "limited"]

    def test_url_decorations_are_stripped(self, norm):
        assert norm("www.eelegal.com")["name_cat"] == "eelegal"

    def test_name_cat_glues_core_tokens(self, norm):
        r = norm("Sunrise Bakery LLC")
        assert r["name_core"] == "sunrise bakery"
        assert r["name_cat"] == "sunrisebakery"
        assert r["legal"] == "llc"

    def test_empty_name_does_not_crash(self, norm):
        r = norm("")
        assert r["core_toks"] == []
        assert r["name_cat"] == ""


class TestAddressNormalisation:
    @pytest.mark.parametrize("raw, expected", [
        ("12 Main Street", "12 main st"),
        ("7 Gandhi Road", "7 gandhi rd"),
        ("9 North Park Avenue", "9 n park av"),
        ("123 Saint Louis Avenue Suite 5", "123 st louis av ste 5"),
        ("10 Rue de la Paix", "10 r de la paix"),
    ])
    def test_abbreviations_are_unified(self, norm, raw, expected):
        assert norm("x", raw)["addr"] == expected

    def test_street_and_st_normalise_identically(self, norm):
        assert norm("x", "12 Main Street")["addr"] == norm("x", "12 Main St")["addr"]

    def test_state_names_become_codes(self, norm):
        assert norm("x", "12 Main Street, Springfield, Illinois")["addr"] == "12 main st springfield il"
        assert norm("x", "New York, New York")["addr"] == "ny ny"

    def test_renamed_indian_cities_are_unified(self, norm):
        assert norm("x", "9 Calcutta Bombay")["addr"] == "9 kolkata mumbai"
        assert norm("x", "Flat 3 Gandhi Nagar Gurugram")["addr"] == "flat 3 gandhi ngr gurgaon"

    @pytest.mark.parametrize("null_like", ["", "<null>", "N/A", "None"])
    def test_null_like_addresses_become_empty(self, norm, null_like):
        r = norm("x", null_like)
        assert r["addr"] == ""
        assert r["nums"] == []
        assert r["addr_toks"] == []

    def test_ordinal_suffixes_are_removed(self, norm):
        assert norm("x", "1st Floor, 22nd Street")["addr"] == "1 flr 22 st"

    def test_letters_glued_to_digits_are_split(self, norm):
        assert norm("x", "221B Baker Street")["addr"] == "221 b baker st"

    def test_transliterated_address(self, norm):
        assert norm("x", "दिल्ली")["addr"] == "dl"


class TestNumbers:
    def test_leading_zeros_are_stripped(self, norm):
        assert norm("x", "0012 Main St")["nums"] == ["12"]
        assert norm("x", "Plot 007, Gandhi Road")["nums"] == ["7"]

    def test_all_zero_number_is_dropped(self, norm):
        assert norm("x", "000")["nums"] == []

    def test_numbers_keep_order_and_are_deduplicated(self, norm):
        assert norm("x", "12 Main St, 12 Main St")["nums"] == ["12"]
        assert norm("x", "Plot No. 5, Sector 12")["nums"] == ["5", "12"]

    def test_digit_glued_to_letter_still_yields_the_number(self, norm):
        assert norm("x", "12A Baker Street")["nums"] == ["12"]

    def test_same_house_number_with_and_without_zero_padding_agree(self, norm):
        assert norm("x", "0042 Elm Rd")["nums"] == norm("x", "42 Elm Road")["nums"]


class TestAddressTokens:
    def test_generic_and_short_words_are_excluded(self, norm):
        # "st", "flr", "plot", "no" are generic; "ny" is below the 3-letter minimum
        assert norm("x", "Plot No. 5, 1st Floor, Main Street, NY")["addr_toks"] == ["main"]

    def test_tokens_are_unique_and_ordered(self, norm):
        assert norm("x", "Baker Street, Baker Street, Elm Road")["addr_toks"] == ["baker", "elm"]


class TestSkeleton:
    def test_devanagari_and_latin_spellings_share_a_skeleton(self, norm):
        a, b = norm("भारत ग्रीन"), norm("bharat green")
        assert a["name_skel"] == b["name_skel"] == "brtgrn"
        assert a["skel_toks"] == b["skel_toks"] == ["brt", "grn"]

    def test_phonetically_close_spellings_collide(self, norm):
        assert norm("Phoenix Tech")["name_skel"] == norm("Fenix Tek")["name_skel"] == "fnkstk"

    def test_different_names_do_not_collide(self, norm):
        assert norm("Green Valley Dental")["name_skel"] != norm("Blue Ridge Plumbing")["name_skel"]

    def test_vowels_after_first_letter_are_dropped_and_repeats_collapse(self, norm):
        assert norm("Aaa Bbb Ccc")["name_skel"] == "abk"
        assert norm("Sharma")["name_skel"] == "srm"

    def test_leading_vowel_is_kept(self):
        out = pl.select(N._skeleton(pl.lit("alpha"))).item()
        assert out == "alf"


class TestIO:
    def test_load_source_reads_tsv_without_quote_handling_and_fills_nulls(self, tmp_path):
        path = tmp_path / "s.tsv"
        path.write_text('business_name\tbusiness_address\tcountry\n'
                        '"Joe\'s" Diner\t1 Main St\tUS\n'
                        'Bakery\t\t\n', encoding="utf-8")
        df = N.load_source(path)
        assert df["business_name"].to_list() == ['"Joe\'s" Diner', "Bakery"]
        assert df["business_address"].to_list() == ["1 Main St", ""]
        assert df["country"].to_list() == ["US", ""]

    def test_load_split_stacks_the_three_sources_with_dense_row_ids(self, tmp_path):
        (tmp_path / "train").mkdir()
        for s, name in enumerate(["Alpha Bakery", "Alpha Bakery Ltd", "Alfa Bakery"], start=1):
            (tmp_path / "train" / f"train_source{s}.tsv").write_text(
                f"business_name\tbusiness_address\tcountry\n{name}\t1 Main St\tUS\n", encoding="utf-8")
        df = N.load_split(tmp_path, "train")
        assert df["rid"].to_list() == [0, 1, 2]
        assert df["src"].to_list() == [1, 2, 3]
        assert df["name_core"].to_list() == ["alpha bakery", "alpha bakery", "alfa bakery"]

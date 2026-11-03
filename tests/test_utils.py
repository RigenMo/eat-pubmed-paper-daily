import ssl
"""Tests for zotero_arxiv_daily.utils: glob_match, send_email, tex extraction."""

import smtplib
import tarfile
import io

import pytest

from zotero_arxiv_daily.utils import glob_match, send_email, extract_tex_code_from_tar
from tests.canned_responses import make_stub_smtp


# ---------------------------------------------------------------------------
# glob_match — migrated from test_glob_match.py
# ---------------------------------------------------------------------------


class TestGlobMatch:
    """Test cases for the glob_match function."""

    def test_exact_match(self):
        assert glob_match("hello.txt", "hello.txt")
        assert not glob_match("hello.txt", "world.txt")
        assert glob_match("", "")

    def test_wildcard_asterisk(self):
        assert glob_match("hello.txt", "*.txt")
        assert not glob_match("hello.py", "*.txt")
        assert glob_match("file", "*")

        assert glob_match("hello.world.txt", "*.*.txt")
        assert not glob_match("hello.txt", "*.*.txt")
        assert glob_match("a.b.c.d", "*.*.*.*")

        assert glob_match("hello_world.txt", "hello*world.txt")
        assert glob_match("hello123world.txt", "hello*world.txt")
        assert glob_match("helloworld.txt", "hello*world.txt")
        assert not glob_match("hello_universe.txt", "hello*world.txt")

    def test_wildcard_question_mark(self):
        assert glob_match("hello.txt", "hell?.txt")
        assert not glob_match("hell.txt", "hell?.txt")
        assert glob_match("hello.txt", "he??o.txt")
        assert glob_match("heXXo.txt", "he??o.txt")
        assert not glob_match("heo.txt", "he??o.txt")

    def test_character_classes(self):
        assert glob_match("file1.txt", "file[123].txt")
        assert glob_match("file2.txt", "file[123].txt")
        assert not glob_match("file4.txt", "file[123].txt")

        assert glob_match("file1.txt", "file[1-3].txt")
        assert glob_match("file2.txt", "file[1-3].txt")
        assert not glob_match("file4.txt", "file[1-3].txt")

        assert glob_match("fileA.txt", "file[!123].txt")
        assert not glob_match("file1.txt", "file[!123].txt")

    def test_path_separators(self):
        assert glob_match("dir/file.txt", "dir/file.txt")
        assert glob_match("dir/file.txt", "*/file.txt")
        assert glob_match("dir/subdir/file.txt", "*/subdir/file.txt")
        assert glob_match("dir/subdir/file.txt", "dir/*/file.txt")
        assert glob_match("dir/subdir/file.txt", "*/*/file.txt")

    def test_complex_patterns(self):
        assert glob_match("test1_file.txt", "test[1-3]*file.txt")
        assert glob_match("test2_long_file.txt", "test[1-3]*file.txt")
        assert not glob_match("test4_file.txt", "test[1-3]*file.txt")

        assert glob_match("prefix_middle_suffix.log", "prefix*middle*.log")
        assert not glob_match("prefix_other_suffix.log", "prefix*middle*.log")

    def test_edge_cases(self):
        assert glob_match("", "**")
        assert not glob_match("", "?")
        assert not glob_match("file", "")

        assert glob_match("file-name.txt", "file-name.txt")
        assert glob_match("file_name.txt", "file_name.txt")
        assert glob_match("file.name.txt", "file.name.txt")

        assert not glob_match("File.txt", "file.txt")
        assert not glob_match("FILE.TXT", "file.txt")
        assert glob_match("file.txt", "file.txt")

    def test_partial_matches(self):
        assert glob_match("hello.txt", "hello.txt")
        assert not glob_match("prefix_hello.txt", "hello.txt")
        assert glob_match("hello.txt", "*.txt")

    def test_special_glob_characters(self):
        assert not glob_match("file[1].txt", "file[1].txt")
        assert glob_match("file1.txt", "file[1].txt")

    def test_numeric_patterns(self):
        assert glob_match("file001.txt", "file???.txt")
        assert not glob_match("file01.txt", "file???.txt")
        assert glob_match("version1.2.3.txt", "version*.txt")
        assert glob_match("version1.2.3.txt", "version?.?.?.txt")

    def test_extension_patterns(self):
        assert glob_match("document.pdf", "*.pdf")
        assert glob_match("image.jpg", "*.jpg")
        assert glob_match("script.py", "*.py")
        assert not glob_match("data.csv", "*.txt")

        assert glob_match("file.txt", "*.[tc][xs][tv]")
        assert glob_match("file.csv", "*.[tc][xs][tv]")
        assert not glob_match("file.pdf", "*.[tc][xs][tv]")

    def test_recursive_wildcard(self):
        assert glob_match("file.txt", "**/*.txt")
        assert glob_match("dir/file.txt", "**/*.txt")
        assert glob_match("dir/subdir/file.txt", "**/*.txt")
        assert glob_match("dir/subdir/subsubdir/file.txt", "**/*.txt")


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------


def test_send_email_starttls_success(config, monkeypatch):
    sent = []
    calls = []

    class StartTLS(make_stub_smtp(sent)):
        def starttls(self, context=None):
            assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
            calls.append("starttls")

    monkeypatch.setattr(smtplib, "SMTP", StartTLS)
    send_email(config, "<html>hello</html>")
    assert calls == ["starttls"]
    assert len(sent) == 1
    sender, recipients, body = sent[0]
    assert sender == "test@example.com"
    assert recipients == ["test@example.com"]
    # Body is a full MIME message (base64-encoded). Check the raw MIME string.
    assert "text/html" in body


def test_send_email_ssl_port_uses_timeout(config, monkeypatch):
    sent = []
    config.email.smtp_port = 465
    stub = make_stub_smtp(sent)
    calls = []

    def smtp_ssl(host, port, timeout, context):
        assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
        calls.append((host, port, timeout))
        return stub()

    monkeypatch.setattr(smtplib, "SMTP_SSL", smtp_ssl)
    monkeypatch.setattr(smtplib, "SMTP", lambda *a, **kw: pytest.fail("unexpected SMTP"))
    send_email(config, "<html>ssl</html>")
    assert calls == [(config.email.smtp_server, 465, 30)]
    assert len(sent) == 1


def test_send_email_tls_failure_does_not_downgrade(config, monkeypatch):
    calls = []

    class TLSFails:
        def __init__(self, host, port, timeout):
            calls.append((host, port, timeout))
        def starttls(self, context=None):
            assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
            raise OSError("TLS failed")
        def close(self):
            pass

    monkeypatch.setattr(smtplib, "SMTP", TLSFails)
    monkeypatch.setattr(smtplib, "SMTP_SSL", lambda *a, **kw: pytest.fail("unexpected fallback"))
    with pytest.raises(OSError, match="TLS failed"):
        send_email(config, "hello")
    assert calls == [(config.email.smtp_server, config.email.smtp_port, 30)]


def test_send_email_quit_failure_after_acceptance(config, monkeypatch):
    sent = []

    class QuitFails(make_stub_smtp(sent)):
        def quit(self):
            raise OSError("connection lost")

    monkeypatch.setattr(smtplib, "SMTP", QuitFails)
    send_email(config, "hello")
    assert len(sent) == 1


@pytest.mark.parametrize("subject, expected", [
    (None, "Eat Paper Daily 2026/11/03"),
    ("Eat Paper Daily 2026/11/03（生成失败）", "Eat Paper Daily 2026/11/03（生成失败）"),
])
def test_send_email_china_date_and_optional_notice_subject(config, monkeypatch, subject, expected):
    import datetime
    from email import message_from_string
    from email.header import decode_header, make_header
    from zotero_arxiv_daily import utils

    utc_time = datetime.datetime(2026, 11, 2, 23, 10, tzinfo=datetime.timezone.utc)
    class FixedDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return utc_time.astimezone(tz) if tz else utc_time.replace(tzinfo=None)

    sent = []
    monkeypatch.setattr(utils.datetime, "datetime", FixedDateTime)
    monkeypatch.setattr(smtplib, "SMTP", make_stub_smtp(sent))
    send_email(config, "hello", subject=subject)
    message = message_from_string(sent[0][2])
    assert str(make_header(decode_header(message["Subject"]))) == expected


# ---------------------------------------------------------------------------
# extract_tex_code_from_tar
# ---------------------------------------------------------------------------


@pytest.fixture()
def make_tar(tmp_path):
    """Create a tar file with given files, auto-cleaned by tmp_path."""

    def _make(files: dict[str, str]) -> str:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for name, content in files.items():
                data = content.encode()
                info = tarfile.TarInfo(name=name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        path = tmp_path / "test.tar"
        path.write_bytes(buf.getvalue())
        return str(path)

    return _make


def test_extract_tex_single_file(make_tar):
    path = make_tar({"main.tex": "\\begin{document}\nHello\n\\end{document}"})
    result = extract_tex_code_from_tar(path, "test-paper")
    assert result is not None
    assert "Hello" in result["all"]


def test_extract_tex_with_input_resolution(make_tar):
    path = make_tar({
        "main.tex": "\\begin{document}\n\\input{intro}\n\\end{document}",
        "main.bbl": "",
        "intro.tex": "This is the introduction.",
    })
    result = extract_tex_code_from_tar(path, "test-paper")
    assert "This is the introduction." in result["all"]


def test_extract_tex_no_tex_files(make_tar):
    path = make_tar({"readme.md": "# Hello"})
    result = extract_tex_code_from_tar(path, "test-paper")
    assert result is None


def test_extract_tex_not_a_tar(tmp_path):
    path = tmp_path / "bad.tar"
    path.write_bytes(b"this is not a tar file")
    result = extract_tex_code_from_tar(str(path), "test-paper")
    assert result is None


def test_extract_tex_multiple_tex_no_bbl(make_tar):
    path = make_tar({
        "a.tex": "\\section{A}",
        "b.tex": "\\begin{document}\nMain content\n\\end{document}",
    })
    result = extract_tex_code_from_tar(path, "test-paper")
    assert result is not None
    assert "Main content" in result["all"]


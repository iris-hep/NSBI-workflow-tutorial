from nsbi_common_utils.test import say_hello

def test_say_hello_prints_message(capsys):
    say_hello()

    captured = capsys.readouterr()
    assert captured.out == "Hello, world!\n"

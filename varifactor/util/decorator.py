def defunct(func):
    def func_wrapper(*args, **kwargs):
        print "varifactor: This method is now defunct and will be removed from later versions"
        return func(*args, **kwargs)

    return func_wrapper


if __name__ == "__main__":
    @defunct
    def get_text(name):
        return "Hello " + name
import asyncio
import functools
import inspect


def async_to_sync(func):
    """将异步方法转换为同步方法的装饰器"""

    def wrapper(*args, **kwargs):
        # 尝试使用Omniverse的事件循环
        try:
            from omni.kit.async_engine import run_coroutine
            from omni.kit.app import get_app

            app = get_app()

            result = run_coroutine(func(*args, **kwargs))
            while not result.done():
                app.update()

            result = result.result()
            return result
        except ImportError:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(func(*args, **kwargs))

    return wrapper


def maybe_coroutine(func):
    """装饰器：处理可能返回协程对象的属性访问"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if inspect.iscoroutine(result):
            # 尝试使用Omniverse的事件循环
            try:
                from omni.kit.async_engine import run_coroutine
                from omni.kit.app import get_app

                app = get_app()

                result = run_coroutine(func(*args, **kwargs))

                while not result.done():
                    app.update()

                result = result.result()
                return result
            except ImportError:
                # 降级到标准asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                return loop.run_until_complete(result)
        return result

    return wrapper

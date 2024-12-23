

<!doctype html>
<html class="h-full overflow-y-scroll">
  <head>
    <title>Ollama</title>

    <meta charset="utf-8" />
    <meta name="description" content="Get up and running with large language models."/>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta property="og:title" content="Ollama" />
    <meta property="og:description" content="Get up and running with large language models." />
    <meta property="og:url" content="https://ollama.com" />
    <meta property="og:image" content="https://ollama.com/public/og.png" />
    <meta property="og:image:type" content="image/png" />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="628" />
    <meta property="og:type" content="website" />

    <meta property="twitter:card" content="summary" />
    <meta property="twitter:title" content="Ollama" />
    <meta property="twitter:description" content="Get up and running with large language models." />
    <meta property="twitter:site" content="ollama" />

    <meta property="twitter:image:src" content="https://ollama.com/public/og-twitter.png" />
    <meta property="twitter:image:width" content="1200" />
    <meta property="twitter:image:height" content="628" />

    <link rel="icon" type="image/png" sizes="16x16" href="/public/icon-16x16.png" />
    <link rel="icon" type="image/png" sizes="32x32" href="/public/icon-32x32.png" />
    <link rel="icon" type="image/png" sizes="48x48" href="/public/icon-48x48.png" />
    <link rel="icon" type="image/png" sizes="64x64" href="/public/icon-64x64.png" />
    <link rel="apple-touch-icon" sizes="180x180" href="/public/apple-touch-icon.png" />
    <link rel="icon" type="image/png" sizes="192x192" href="/public/android-chrome-icon-192x192.png" />
    <link rel="icon" type="image/png" sizes="512x512" href="/public/android-chrome-icon-512x512.png" />

    

    <link href="/public/tailwind.css?v=000d8341e0fc9196c23018df3b98885d" rel="stylesheet" />
    <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "WebSite",
        "name": "Ollama",
        "url": "https://ollama.com"
      }
    </script>

    <script type="text/javascript">
      function copyToClipboard(element) {
        let commandElement = null;
        const preElement = element.closest('pre');
        const languageNoneElement = element.closest('.language-none');

        if (preElement) {
          commandElement = preElement.querySelector('code');
        } else if (languageNoneElement) {
          commandElement = languageNoneElement.querySelector('.command');
        }

        if (!commandElement) {
          console.error('No code or command element found');
          return;
        }

        const code = commandElement.textContent ? commandElement.textContent.trim() : commandElement.value;

        navigator.clipboard
          .writeText(code)
          .then(() => {
            const copyIcon = element.querySelector('.copy-icon')
            const checkIcon = element.querySelector('.check-icon')

            copyIcon.classList.add('hidden')
            checkIcon.classList.remove('hidden')

            setTimeout(() => {
              copyIcon.classList.remove('hidden')
              checkIcon.classList.add('hidden')
            }, 2000)
          })
      }
    </script>
    
    <script>
      
      function getIcon(url) {
        url = url.toLowerCase();
        if (url.includes('x.com') || url.includes('twitter.com')) return 'x';
        if (url.includes('github.com')) return 'github';
        if (url.includes('linkedin.com')) return 'linkedin';
        if (url.includes('youtube.com')) return 'youtube';
        if (url.includes('hf.co') || url.includes('huggingface.co') || url.includes('huggingface.com')) return 'hugging-face';
        return 'default';
      }

      function setInputIcon(input) {
        const icon = getIcon(input.value);
        const img = input.previousElementSibling.querySelector('img');
        img.src = `/public/social/${icon}.svg`;
        img.alt = `${icon} icon`;
      }

      function setDisplayIcon(imgElement, url) {
        const icon = getIcon(url);
        imgElement.src = `/public/social/${icon}.svg`;
        imgElement.alt = `${icon} icon`;
      }
    </script>
    
    <script src="/public/vendor/htmx/bundle.js"></script>
    
  </head>

  <body
    class="
      antialiased
      min-h-screen
      w-full
      m-0
      flex
      flex-col
    "
    hx-on:keydown="
      if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
        // Ignore key events in input fields.
        return;
      }
      if ((event.metaKey && event.key === 'k') || event.key === '/') {
        event.preventDefault();
        const sp = htmx.find('#search');
        sp.focus();
        return;
      }
    "
  >
  <header class="sticky top-0 z-40 bg-white underline-offset-4 lg:static">
    <nav class="flex w-full items-center justify-between px-6 py-3.5 gap-x-2">
      <a href="/" class="z-50">
        <img src="/public/ollama.png" class="w-8" alt="Ollama" />
      </a>
      
      
      <div class="hidden lg:flex items-center space-x-8 ml-8 mr-4 text-lg">
        <a class="hover:underline" href="/blog">Blog</a>
        <a class="hover:underline" target="_blank" href="https://discord.com/invite/ollama">Discord</a>
        <a class="hover:underline" target="_blank" href="https://github.com/ollama/ollama">GitHub</a>
      </div>
  
      
      <div class="flex-grow justify-center hidden lg:flex">
        <div class="relative w-full max-w-xs lg:max-w-[32rem]"
          hx-on:keydown="
            if (event.key === 'Escape') {
              const sp = htmx.find('#searchpreview');
              sp.value = '';
              sp.focus();
              htmx.addClass('#searchpreview', 'hidden');
              return;
            }
            htmx.removeClass('#searchpreview', 'hidden');
          "
        >
          
<div 
class="relative flex w-full appearance-none border border-neutral-200 items-center rounded-lg transition-all duration-300 ease-in-out transform focus-within:shadow-sm [transition:box-shadow_0s]"
hx-on:focusout="
  if (!this.contains(event.relatedTarget)) {
    htmx.addClass('#searchpreview', 'hidden');
  }
"
>
<span id="searchIcon" class="pl-2 text-2xl text-neutral-500">
  <svg class="mt-0.25 ml-1 h-4 w-4 fill-current" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
    <path d="m8.5 3c3.0375661 0 5.5 2.46243388 5.5 5.5 0 1.24832096-.4158777 2.3995085-1.1166416 3.3225711l4.1469717 4.1470988c.2928932.2928932.2928932.767767 0 1.0606602-.2662666.2662665-.6829303.2904726-.9765418.0726181l-.0841184-.0726181-4.1470988-4.1469717c-.9230626.7007639-2.07425014 1.1166416-3.3225711 1.1166416-3.03756612 0-5.5-2.4624339-5.5-5.5 0-3.03756612 2.46243388-5.5 5.5-5.5zm0 1.5c-2.209139 0-4 1.790861-4 4s1.790861 4 4 4 4-1.790861 4-4-1.790861-4-4-4z" />
  </svg>
</span>
<form action="/search" autocomplete="off" class="w-full">
  <input
    id="search"
    hx-get="/search"
    hx-trigger="keyup changed delay:100ms, focus"
    hx-target="#searchpreview"
    hx-swap="innerHTML"
    name="q"
    class="resize-none rounded-lg border-0 py-2.5 pr-10 text-sm w-full focus:outline-none focus:ring-0 transition-shadow duration-300 ease-in-out"
    placeholder="Search models"
    autocomplete="off"
    hx-on:keydown="
      if (event.key === 'Enter') {
        event.preventDefault();
        window.location.href = '/search?q=' + encodeURIComponent(this.value);
        return;
      }
      if (event.key === 'Escape') {
        event.preventDefault();
        this.value = '';
        this.blur();
        htmx.addClass('#searchpreview', 'hidden');
        return;
      }
      htmx.removeClass('#searchpreview', 'hidden');
    "
    hx-on:focus="
      htmx.removeClass('#searchpreview', 'hidden')
    "
  />
</form>
<div id="searchpreview" class="hidden absolute left-0 right-0 top-12 z-50" style="width: calc(100% + 2px); margin-left: -1px;"></div>
</div>

        </div>
      </div>
  
      
      <div class="hidden lg:flex items-center space-x-8 ml-4 text-lg">
        <a class="hover:underline" href="/models">Models</a>
        
        <div class="flex-none">
          <div 
            class="relative"
            hx-on:focusout="
              if (!this.contains(event.relatedTarget)) {
                htmx.addClass('#user-nav', 'hidden');
              }
            "
          >
            
              <a href="/signin" class="block whitespace-nowrap hover:underline">Sign in</a>
            
          </div>
        </div>
        
          <a class="flex cursor-pointer items-center rounded-lg bg-neutral-800 px-4 py-1 text-white hover:bg-black" href="/download">
            Download
          </a>
        
      </div>
  
      
      <div class="lg:hidden flex items-center">
        <input type="checkbox" id="menu" class="peer hidden" />
        <label for="menu" class="z-50 cursor-pointer peer-checked:hidden block">
          <svg
            class="h-8 w-8"
            fill="none"
            viewBox="0 0 24 24"
            stroke-width="1.5"
            stroke="currentColor"
            aria-hidden="true"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"
            />
          </svg>
        </label>
        <label for="menu" class="z-50 cursor-pointer hidden peer-checked:block fixed top-4 right-6">
          <svg
            class="h-8 w-8"
            fill="none"
            viewBox="0 0 24 24"
            stroke-width="1.5"
            stroke="currentColor"
            aria-hidden="true"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </label>
        
        <div class="fixed inset-0 bg-white z-40 hidden peer-checked:block overflow-y-auto">
          <div class="flex flex-col space-y-6 px-6 py-4 pt-24 text-3xl tracking-tight">
            <a href="/models">Models</a>
            <a href="https://discord.com/invite/ollama">Discord</a>
            <a href="/blog">Blog</a>
            <a href="https://github.com/ollama/ollama">GitHub</a>
            <a href="/download">Download</a>
        
            
            <a href="/signin" class="block">Sign in</a>
            
          </div>
        </div>
      </div>
    </nav>
  </header>

    <main class="flex-grow">
      <main class="mx-auto flex max-w-4xl flex-1 flex-col-reverse items-center justify-center p-32 md:flex-row md:items-start md:justify-between">
  <div class="space-y-2 text-center md:pt-6 md:text-left">
    <h2 class="text-3xl font-normal tracking-tight md:text-4xl">
      404.
      <span class="text-neutral-400"> That's an error. </span>
    </h2>
    <p class="text-center text-lg md:text-left md:text-xl">
      The page was not found.
    </p>
  </div>
  <div class="pb-4 md:pb-0">
    <img src="/public/400s.svg" class="w-40 md:w-48" alt="400s ollama" />
  </div>
</main>
    </main>

    <footer class="mt-auto">
      <div class="bg-white underline-offset-4 hidden md:block">
        <div class="flex items-center justify-between px-6 py-3.5">
          <div class="text-xs text-neutral-500">© 2024 Ollama</div>
          <div class="flex space-x-6 text-xs text-neutral-500">
            <a href="/blog" class="hover:underline">Blog</a>
            <a href="https://github.com/ollama/ollama/tree/main/docs" class="hover:underline">Docs</a>
            <a href="https://github.com/ollama/ollama" class="hover:underline">GitHub</a>
            <a href="https://discord.com/invite/ollama" class="hover:underline">Discord</a>
            <a href="https://twitter.com/ollama" class="hover:underline">X (Twitter)</a>
            <a href="https://lu.ma/ollama" class="hover:underline">Meetups</a>
          </div>
        </div>
      </div>
      <div class="bg-white py-4 md:hidden">
        <div class="flex flex-col items-center justify-center">
          <ul class="flex flex-wrap items-center justify-center text-sm text-neutral-500">
            <li class="mx-2 my-1">
              <a href="/blog" class="hover:underline">Blog</a>
            </li>
            <li class="mx-2 my-1">
              <a href="https://github.com/ollama/ollama/tree/main/docs" class="hover:underline">Docs</a>
            </li>
            <li class="mx-2 my-1">
              <a href="https://github.com/ollama/ollama" class="hover:underline">GitHub</a>
            </li>
          </ul>
          <ul class="flex flex-wrap items-center justify-center text-sm text-neutral-500">
            <li class="mx-2 my-1">
              <a href="https://discord.com/invite/ollama" class="hover:underline">Discord</a>
            </li>
            <li class="mx-2 my-1">
              <a href="https://twitter.com/ollama" class="hover:underline">X (Twitter)</a>
            </li>
            <li class="mx-2 my-1">
              <a href="https://lu.ma/ollama" class="hover:underline">Meetups</a>
            </li>
          </ul>
          <div class="mt-2 flex items-center justify-center text-sm text-neutral-500">
            © 2024 Ollama Inc.
          </div>
        </div>
      </div>
    </footer>

    
    <span class="hidden" id="end_of_template"></span>
  </body>
</html>
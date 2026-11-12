r"""Check onboarding layout and interactions in the running Gradio application.

Install browser tooling once:
    python3 -m pip install playwright
    python3 -m playwright install chromium webkit

Check the deployed Space:
    python3 scripts/check_space_ui.py \
        --url https://chrisvoncsefalvay-vespasian.hf.space \
        --theme light --output runs/space-ui/light

Check its Hugging Face iframe:
    python3 scripts/check_space_ui.py \
        --url https://huggingface.co/spaces/chrisvoncsefalvay/vespasian \
        --embed --sizes 390x844,1280x900 --output runs/space-ui/embedded

Screenshots and computed styles are saved with a JSON receipt. Experiment,
GPU advance and Hub publication requests are blocked and fail the check.
Ordinary Gradio load callbacks remain enabled. Local fake Space metadata is
supplied only when localhost identifies itself as test/vespasian.
"""
import argparse
import json
import re
import traceback
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, expect

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--url', default='https://chrisvoncsefalvay-vespasian.hf.space')
parser.add_argument('--output', required=True)
parser.add_argument('--engines', default='chromium,webkit')
parser.add_argument('--embed', action='store_true')
parser.add_argument('--sizes', default='390x600,320x560,430x740,1280x900,844x390')
parser.add_argument('--theme', default='dark', choices=['dark', 'light'])
args = parser.parse_args()
engines = args.engines.split(',')
if any(engine not in {'chromium', 'webkit'} for engine in engines):
    parser.error('--engines accepts chromium,webkit')
try:
    sizes = [tuple(map(int, size.split('x'))) for size in args.sizes.split(',')]
    if any(len(size) != 2 or min(size) < 200 for size in sizes):
        raise ValueError
except ValueError:
    parser.error('--sizes requires widthxheight values of at least 200px')
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
result = {'url': args.url, 'embed': args.embed, 'theme': args.theme, 'browsers': {}, 'cases': [], 'interactions': [], 'errors': [], 'forbidden': []}


def blocked_functions(config):
    sources = {
        component['id'] for component in config['components']
        if component.get('props', {}).get('elem_id') == 'vs-run-button'
        or component.get('props', {}).get('value') in (
            'Save edited model to Hub', 'Save trajectory to Hub',
            'Save latest/best patches privately',
        )
    }
    return {
        dep['id'] for dep in config['dependencies']
        if dep.get('api_name') in ('run', 'advance', 'publish', 'publish_run')
        or any(target[0] in sources for target in dep.get('targets', []))
    }

MEASURE = '''() => {
 const dialog=document.querySelector('#vs-guide'), body=dialog.querySelector('.vs-guide-body'), footer=dialog.querySelector('.vs-guide-footer');
 const rect=el=>{const r=el.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom,right:r.right}};
 const style=el=>{const s=getComputedStyle(el);return {fontSize:s.fontSize,fontFamily:s.fontFamily,lineHeight:s.lineHeight,color:s.color,backgroundColor:s.backgroundColor,padding:s.padding,display:s.display,overflow:s.overflow,opacity:s.opacity}};
 const rgb=value=>(value.match(/[\\d.]+/g)||[]).map(Number);
 const luminance=channels=>channels.slice(0,3).map(c=>c/255).map(c=>c<=.04045?c/12.92:((c+.055)/1.055)**2.4).reduce((a,c,i)=>a+c*[.2126,.7152,.0722][i],0);
 const contrast=el=>{let node=el,bg; while(node){const c=rgb(getComputedStyle(node).backgroundColor);if(c.length===3||c[3]===1){bg=c;break}node=node.parentElement}const fg=luminance(rgb(getComputedStyle(el).color)),back=luminance(bg||[255,255,255]);return (Math.max(fg,back)+.05)/(Math.min(fg,back)+.05)};
 const controls=[...dialog.querySelectorAll('button')].filter(el=>el.getClientRects().length).map(el=>({text:el.textContent,disabled:el.disabled,rect:rect(el),style:style(el),contrast:contrast(el)}));
 const tabs=dialog.querySelector('.vs-guide-tabs');
 return {viewport:{width:innerWidth,height:innerHeight},dialog:{rect:rect(dialog),style:style(dialog),scrollHeight:dialog.scrollHeight,clientHeight:dialog.clientHeight,scrollWidth:dialog.scrollWidth,clientWidth:dialog.clientWidth},body:{rect:rect(body),style:style(body),scrollHeight:body.scrollHeight,clientHeight:body.clientHeight,scrollTop:body.scrollTop},footer:{rect:rect(footer),style:style(footer)},title:{rect:rect(document.querySelector('#vs-guide-title')),style:style(document.querySelector('#vs-guide-title'))},description:{rect:rect(document.querySelector('#vs-guide-description')),style:style(document.querySelector('#vs-guide-description'))},tabs:{rect:rect(tabs),items:[...tabs.children].map(rect)},controls,documentWidth:document.documentElement.scrollWidth};
}'''

def checks(m):
    failures = []
    def check(ok, text):
        if not ok:
            failures.append(text)
    check(m['dialog']['scrollWidth'] <= m['dialog']['clientWidth'] + 1, 'Dialog has horizontal overflow')
    check(m['documentWidth'] <= m['viewport']['width'] + 1, 'Document has horizontal overflow')
    check(float(m['title']['style']['fontSize'].removesuffix('px')) >= 26, 'Heading is smaller than 26px')
    check(m['description']['rect']['y'] - m['title']['rect']['bottom'] >= 12, 'Heading and description need at least 12px spacing')
    check(m['footer']['rect']['bottom'] <= min(m['dialog']['rect']['bottom'], m['viewport']['height']) + 1, 'Footer is clipped or requires outer-dialog scrolling')
    check(m['dialog']['rect']['y'] >= -1 and m['dialog']['rect']['bottom'] <= m['viewport']['height'] + 1, 'Dialog is outside the viewport')
    check(m['tabs']['rect']['height'] <= 49, 'Guide navigation is taller than a single 48px row')
    check(max(tab['y'] for tab in m['tabs']['items']) - min(tab['y'] for tab in m['tabs']['items']) <= 1, 'Guide navigation wraps onto multiple rows')
    for c in m['controls']:
        check(c['rect']['height'] >= 43 and c['rect']['width'] >= 43, 'Control is smaller than 44px: ' + c['text'])
        if not c['disabled']:
            check(c['contrast'] >= 4.5, 'Button label contrast is below 4.5:1: ' + c['text'])
    return failures

with sync_playwright() as p:
    for engine in engines:
        browser = None
        try:
            browser = getattr(p, engine).launch(headless=True)
            result['browsers'][engine] = browser.version
            for width, height in sizes:
                context = browser.new_context(viewport={'width': width, 'height': height}, device_scale_factor=1, reduced_motion='reduce', color_scheme=args.theme, is_mobile=width<500, has_touch=width<500)
                context.add_init_script('localStorage.setItem("vespasian.theme", ' + json.dumps(args.theme) + ')')
                page = context.new_page()
                page.set_default_timeout(10000)
                page.on('pageerror', lambda e: result['errors'].append(str(e)))
                configurations = {}

                def configuration(origin):
                    if origin not in configurations:
                        response = context.request.get(origin + '/config')
                        if not response.ok:
                            raise RuntimeError(f'Cannot read {origin}/config: HTTP{response.status}')
                        configurations[origin] = response.json()
                    return configurations[origin]

                def guard(route):
                    request = route.request
                    try:
                        body = request.post_data_json or {}
                    except Exception:
                        body = {}
                    url = urlparse(request.url)
                    origin = f'{url.scheme}://{url.netloc}'
                    forbidden_functions = blocked_functions(configuration(origin))
                    if request.method == 'POST' and (body.get('fn_index') in forbidden_functions or re.search(r'/(call|run)/(run|advance|publish)', request.url)):
                        result['forbidden'].append(request.url)
                        route.abort()
                    else:
                        route.continue_()
                page.route(re.compile(r'/gradio_api/(queue/join|run/|call/)'), guard)
                if urlparse(args.url).hostname in {'127.0.0.1', 'localhost'}:
                    origin = args.url.rstrip('/')
                    if configuration(origin).get('space_id') == 'test/vespasian':
                        result['local_metadata_fixture'] = True
                        page.route('https://huggingface.co/api/spaces/test/vespasian', lambda route: route.fulfill(json={'id':'test/vespasian','author':'test','sha':'fixture','cardData':{'title':'Vespasian'},'likes':0}))
                page.goto(args.url, wait_until='domcontentloaded', timeout=90000)
                frame = page
                if args.embed:
                    page.wait_for_selector('iframe[src*="hf.space"]', timeout=90000)
                    iframe = page.locator('iframe[src*="hf.space"]').first
                    for attempt in range(3):
                        try:
                            iframe.scroll_into_view_if_needed()
                            break
                        except Exception:
                            if attempt == 2:
                                raise
                            page.wait_for_timeout(1000)
                    frame = iframe.content_frame
                expect(frame.locator('#vs-guide[open]')).to_be_visible(timeout=90000)
                expect(frame.locator('html')).to_have_attribute('data-vs-theme', args.theme)
                for index, name in enumerate(['welcome', 'weight-editing', 'how-to-use', 'quick-starters']):
                    frame.locator(f'[data-guide-tab="{index}"]').click()
                    page.wait_for_timeout(250)
                    metrics = frame.locator('#vs-guide').evaluate('(el, fn) => (new Function("return (" + fn + ")()"))()', MEASURE)
                    case = {'engine':engine,'width':width,'height':height,'section':name,'metrics':metrics,'failures':checks(metrics)}
                    screenshot = f'{engine}-{width}x{height}-{name}.png'
                    page.screenshot(path=str(out/screenshot), full_page=False)
                    case['screenshot'] = screenshot
                    if index == 3:
                        before = metrics['footer']['rect']['y']
                        frame.locator('.vs-guide-body').evaluate('(el)=>el.scrollTop=el.scrollHeight')
                        after = frame.locator('.vs-guide-footer').evaluate('(el)=>el.getBoundingClientRect().y')
                        if abs(before-after) > 1:
                            case['failures'].append('Footer moves when the reading area scrolls')
                        page.screenshot(path=str(out/f'{engine}-{width}x{height}-quick-starters-bottom.png'), full_page=False)
                    result['cases'].append(case)
                    (out/'receipt.json').write_text(json.dumps(result, indent=2))
                page.keyboard.press('Escape')
                expect(frame.locator('#vs-guide')).not_to_be_visible()
                trigger = frame.locator('[data-vs-help]').first
                trigger.click()
                expect(frame.locator('#vs-guide[open]')).to_be_visible()
                page.keyboard.press('Escape')
                try:
                    expect(trigger).to_be_focused(timeout=2000)
                    result['interactions'].append(f'{engine}/{width}x{height}: Escape, reopen and focus restoration')
                except Exception:
                    result['errors'].append(f'{engine}/{width}x{height}: Guide does not restore focus to the help trigger')
                if frame.locator('[data-vs-theme-toggle]').count():
                    toggle = frame.locator('[data-vs-theme-toggle]').first
                    toggle.click()
                    opposite = 'light' if args.theme == 'dark' else 'dark'
                    expect(frame.locator('html')).to_have_attribute('data-vs-theme', opposite)
                    trigger.click()
                    expect(frame.locator('#vs-guide[open]')).to_be_visible()
                    page.keyboard.press('Escape')
                    toggle.click()
                    expect(frame.locator('html')).to_have_attribute('data-vs-theme', args.theme)
                    result['interactions'].append(f'{engine}/{width}x{height}: Theme toggle works in both directions with guide reopen')
                context.close()
        except Exception:
            result['errors'].append(engine + ': ' + traceback.format_exc())
        finally:
            if browser:
                browser.close()

result['failed_cases'] = sum(bool(case['failures']) for case in result['cases'])
result['passed_cases'] = len(result['cases']) - result['failed_cases']
(out/'receipt.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'passed_cases':result['passed_cases'],'failed_cases':result['failed_cases'],'errors':result['errors'],'forbidden':result['forbidden']}, indent=2))
raise SystemExit(1 if result['failed_cases'] or result['errors'] or result['forbidden'] else 0)

#!/usr/bin/env bun
/**
 * Reddit reader — Bun's BoringSSL TLS bypasses Cloudflare detection.
 *
 * Usage:
 *   reddit.ts search <query> [--limit N]
 *   reddit.ts sub [query] [--limit N]       (alias for search)
 *   reddit.ts r <subreddit> [--sort hot|new|top|rising] [--limit N]
 *   reddit.ts post <post_id>                (view single post + comments)
 */
const UAS = [
  "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:128.0) Gecko/20100101 Firefox/128.0",
];

function hdr() {
  return {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1", "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document", "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none", "Sec-Fetch-User": "?1",
    "Pragma": "no-cache", "Cache-Control": "no-cache",
  };
}

function escape(s: string) {
  return s.replace(/&amp;/g,"&").replace(/&lt;/g,"<").replace(/&gt;/g,">")
          .replace(/&quot;/g,'"').replace(/&#x27;/g,"'").replace(/&#x2F;/g,"/")
          .replace(/&#x200B;/g,"").replace(/&#32;/g," ");
}

interface Post { title: string; score: string; comments: string; subreddit: string; url: string; }

async function get(url: string, retries=2): Promise<string> {
  for (let i=0;i<=retries;i++) {
    try {
      const r=await fetch(url,{headers:{"User-Agent":UAS[i%UAS.length],...hdr()},signal:AbortSignal.timeout(15e3)});
      if (r.ok||r.status===403||r.status===429) return await r.text();
      await new Promise(r=>setTimeout(r,1e3*(i+1)));
    } catch { if(i<retries) await new Promise(r=>setTimeout(r,1e3*(i+1))); else return ""; }
  }
  return "";
}

function parsePage(html: string): Post[] {
  const posts: Post[] = [], seen = new Set<string>();
  // search-title links (search results)
  let re = /<a[^>]*href="([^"]+)"[^>]*class="[^"]*search-title[^"]*"[^>]*>(.*?)<\/a>/gi;
  let m: RegExpExecArray|null;
  while ((m=re.exec(html))) {
    const t=escape(m[2].replace(/<[^>]+>/g,"").trim()), u=m[1].startsWith("http")?m[1]:`https://old.reddit.com${m[1]}`;
    if(seen.has(u)) continue; seen.add(u);
    const s=m[1].match(/\/r\/(\w+)/i);
    posts.push({title:t,score:"?",comments:"?",subreddit:s?s[1]:"?",url:u});
  }
  // thing elements (subreddit listings)
  re = /<div[^>]*id="thing_(t3_\w+)"[^>]*>.*?<a[^>]*class="[^"]*title[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)<\/a>/gi;
  while ((m=re.exec(html))) {
    const t=escape(m[3].replace(/<[^>]+>/g,"").trim()), u=m[2].startsWith("http")?m[2]:`https://old.reddit.com${m[2]}`;
    if(seen.has(u)) continue; seen.add(u);
    const s=m[2].match(/\/r\/(\w+)/i);
    posts.push({title:t,score:"?",comments:"?",subreddit:s?s[1]:"?",url:u});
  }
  return posts;
}

export async function search(q: string, lim=5): Promise<Post[]> {
  return parsePage(await get(`https://old.reddit.com/search?q=${encodeURIComponent(q)}&limit=${lim}&sort=new&t=all`)).slice(0,lim);
}

export async function subreddit(name: string, sort="hot", lim=5): Promise<Post[]> {
  return parsePage(await get(`https://old.reddit.com/r/${name}/${sort}/`)).slice(0,lim);
}

export interface Comment { author: string; body: string; score: string; }
export interface PostDetail { title: string; score: string; comments_count: string; body: string; url: string; comments: Comment[]; }

export async function postDetail(postId: string): Promise<PostDetail|null> {
  const html = await get(`https://old.reddit.com/comments/${postId}/`);
  if (!html) return null;

  const titleM = html.match(/<a[^>]*class="[^"]*title[^"]*"[^>]*>(.*?)<\/a>/);
  const title = titleM ? escape(titleM[1].replace(/<[^>]+>/g,"").trim()) : "?";

  const scoreM = html.match(/class="[^"]*score[^"]*"[^>]*unvoted[^>]*>(-?\d+)</);
  const score = scoreM ? scoreM[1] : "?";

  // Extract selftext
  const bodyM = html.match(/<div class="md"><p>(.*?)<\/p><\/div>/);
  const body = bodyM ? escape(bodyM[1].replace(/<[^>]+>/g," ").trim()).substring(0,1000) : "";

  // Extract comments
  const comments: Comment[] = [];
  const cmtRe = /<div[^>]*class="[^"]*entry[^"]*"[^>]*>.*?<a[^>]*class="[^"]*author[^"]*"[^>]*>([^<]+)<.*?<div class="md"><p>(.*?)<\/p><\/div>.*?<span class="score[^"]*">(\d+) points?/gis;
  let cm: RegExpExecArray|null;
  while ((cm=cmtRe.exec(html))) {
    comments.push({author:escape(cm[1].trim()),body:escape(cm[2].replace(/<[^>]+>/g," ").trim()).substring(0,500),score:cm[3]});
  }

  const cmCntM = html.match(/(\d+)\s*comments?/i);
  return { title, score, comments_count: cmCntM?cmCntM[1]:"?", body, url: `https://redd.it/${postId}`, comments };
}

async function main() {
  const cmd=process.argv[2], args=process.argv.slice(3);
  let lim=5,sort="hot",fa:string[]=[];
  for(let i=0;i<args.length;i++){ if(args[i]==="--limit"&&i+1<args.length) lim=parseInt(args[++i]); else if(args[i]==="--sort"&&i+1<args.length) sort=args[++i]; else fa.push(args[i]); }
  let p:Post[]=[];
  if(cmd==="search"||cmd==="sub") p=await search(fa.join(" "),lim);
  else if(cmd==="r") p=await subreddit(fa[0]||"LocalLLaMA",sort,lim);
  else if(cmd==="post") {
    const d=await postDetail(fa[0]||"");
    if(!d){console.log("Post not found");process.exit(1);}
    console.log(`▲${d.score} | ${d.comments_count} comments`);
    console.log(`  ${d.title}`);
    if(d.body) console.log(`\n  ${d.body}\n`);
    for(const c of d.comments.slice(0,5)) console.log(`  [${c.score}] ${c.author}: ${c.body.substring(0,200)}\n`);
    return;
  } else { console.log("Usage:\n  reddit.ts search|sub <query> [--limit N]\n  reddit.ts r <subreddit> [--sort hot|new|top|rising] [--limit N]\n  reddit.ts post <post_id>"); process.exit(1); }
  for(const x of p) console.log(`r/${x.subreddit} | ▲${x.score} | ${x.comments}cmts\n  ${x.title}\n  ${x.url}\n`);
}
main();

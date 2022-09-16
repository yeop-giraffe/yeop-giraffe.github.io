---
layout: post
title:  "Upload folders on GitHub"
summary: folder upload on github
author: yeop-giraffe
date: '2022-09-16'
category: [Coding, etc.]
tag: github
thumbnail: 
---

# Steps
OS : window 10 <br/>

## 1. 업로드할 폴더의 상위 폴더에서 "Git Bash Here" 클릭
## 2. 명령어 입력
```terminal
$ git init
$ git status
$ git add .
$ git commit -m "commit log"
$ git remote -v
$ git push origin main
```
## 2-1. (Error) Author identity unknown 
git commit -m "commit log"에서 발생<br/>
아래 명령어 중 하나 입력
```terminal
$ git config --global user.email "email address"
$ git config --global user.name "user name"
```

## 2-2. (Error) fatal: 'origin' does not appear to be a git repository
Git push origin main에서 발생<br/>
```terminal
$ git remote rm origin  //연결 해제
$ git remote add origin "git repository address"  //재연결
$ git push origin main  //reject 오류: git push origin +main
```






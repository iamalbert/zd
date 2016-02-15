package = "zd"
version = "scm-1"
source = {
   url = "git@github.com:iamalbert/zd.git",
   tag = "master"
}
description = {
   summary = "my deep learning framework",
   detailed = [[
        my deep learning framework
   ]],
   homepage = "git@github.com:iamalbert/zd.git",
   license = "MIT"
}
dependencies = {
   "torch >= 7.0"
}
build = {
   type = "make",
   build_variables = {
      CFLAGS = "-std=c99 -Wall -pedantic -O2 -I$(LUA_INCDIR)/TH -I$(LUA_INCDIR)",
      LIBFLAG = "$(LIBFLAG)",
      LUA = "$(LUA)",
      LUA_BINDIR = "$(LUA_BINDIR)",
      LUA_INCDIR = "$(LUA_INCDIR)",
      LUA_LIBDIR = "$(LUA_LIBDIR)"
   },
   install_variables = {
      BINDIR = "$(BINDIR)",
      CONFDIR = "$(CONFDIR)",
      LIBDIR = "$(LIBDIR)",
      LUADIR = "$(LUADIR)",
      PREFIX = "$(PREFIX)"
   }
}

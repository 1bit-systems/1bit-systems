for %i run (0 5)
  fs%i:
  if exist "SmokelessRuntimeEFIPatcher(020).efi" then
	SmokelessRuntimeEFIPatcher(020).efi ENG
	goto END
  endif
endfor

:END